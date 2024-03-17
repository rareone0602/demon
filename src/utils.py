import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as transforms
from tqdm import tqdm, trange

DEVICE = torch.device("cuda")
FILE_PATH = "stablediffusionapi/anything-v5"
tokenizer = CLIPTokenizer.from_pretrained(FILE_PATH, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(FILE_PATH, subfolder="text_encoder").to(DEVICE)
vae = AutoencoderKL.from_pretrained(FILE_PATH, subfolder='vae').to(DEVICE)

def from_pil_to_latent(pil_img):
    # Convert the PIL Image to a tensor
    pil_img = pil_img.resize((512, 512))
    transformer = transforms.ToTensor()
    img = transformer(pil_img)

    # Add the batch dimension
    img = img.unsqueeze(0).to(vae.device)

    return encode(img)

def from_latent_to_pil(latent):
    # Decode the latent tensor
    img = decode(latent)

    # Convert the tensor to a PIL Image
    pil_img = to_pil(img)

    return pil_img

def get_embedding(text):
    with torch.no_grad():
        input_ids = tokenizer(text, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        input_ids = input_ids.to(DEVICE)
        text_embedding = text_encoder(input_ids)[0]
    return text_embedding

def encode(input_img): # Autoencoder takes [-1, 1] as input
    if len(input_img.shape)<4:
        input_img = input_img.unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode(input_img * 2 - 1) 
    return vae.config.scaling_factor * latent.latent_dist.sample()

def decode(latents):
    latents = (1 / vae.config.scaling_factor) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach()
    return image

def to_pil(scaled_img):
    # Remove the batch dimension
    scaled_img = scaled_img.squeeze(0)

    # Convert the tensor to a PIL Image
    pil_transformer = transforms.ToPILImage()
    pil_img = pil_transformer(scaled_img)

    return pil_img

def sde_step(func, x, z, prev_t, t, text_embedding, cfg=2):
    
    about_batch_size = 4
    
    # chunk the input tensor to avoid OOM
    xs = torch.chunk(x, x.size(0) // about_batch_size)
    zs = torch.chunk(z, z.size(0) // about_batch_size)
    ret_x = []
    dt = t - prev_t
    
    for x, z in zip(xs, zs):
        rand_term = z * torch.sqrt(torch.abs(dt))
        f1, g1 = func(prev_t, x, text_embedding, cfg=cfg)
        x_pred = x + f1 * dt + g1 * rand_term
        f2, g2 = func(t, x_pred, text_embedding, cfg=cfg)
        ret_x.append(x + 0.5 * (f1 + f2) * dt + 0.5 * (g1 + g2) * rand_term)
    
    return torch.cat(ret_x, dim=0)
    

def sdeint_variant(func, x, ts, text_embedding, start_time=1., cfg=2):
    if ts.requires_grad:
        raise ValueError("The tensor 'ts' should not require gradients.")
    
    first_z = torch.randn_like(x)
    x = sde_step(func, x, first_z, start_time, ts[0], text_embedding, cfg=cfg)
    
    prev_t = ts[0]

    for t in tqdm(ts[1:]):
        z = torch.randn_like(x)
        x = sde_step(func, x, z, prev_t, t, text_embedding, cfg=cfg)

        prev_t = t

    return first_z, x



def sdeint(func, x, ts, prompts, start_time=1.):
    if ts.requires_grad:
        raise ValueError("The tensor 'ts' should not require gradients.")
    prev_t = start_time
    ans = []
    noises = []
    empty_text_embedding = get_embedding("")

    def get_f_g(t, x):
        f_emp, g_emp = func(t, x, empty_text_embedding)
        f1 = 0
        for text_embedding, cfg in prompts:
            f, _ = func(t, x, text_embedding)
            f1 += (f - f_emp) * cfg
        return f_emp + f1, g_emp

    for t in tqdm(ts):
        dt = t - prev_t
        z = torch.randn_like(x)
        rand_term = z * torch.sqrt(torch.abs(dt))
        
        f1, g1 = get_f_g(prev_t, x)
        x_pred = x + f1 * dt + g1 * rand_term
        f2, g2 = get_f_g(t, x_pred)
        x = x + 0.5 * (f1 + f2) * dt + 0.5 * (g1 + g2) * rand_term

        prev_t = t

        ans.append(x)
        noises.append(z)

    return ans, noises


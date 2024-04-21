import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
import torchvision.transforms as transforms
from tqdm import tqdm, trange
from config import FILE_PATH, DTYPE, DEVICE

tokenizer = CLIPTokenizer.from_pretrained(FILE_PATH, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(FILE_PATH, subfolder="text_encoder").to(device=DEVICE, dtype=DTYPE)
vae = AutoencoderKL.from_pretrained(FILE_PATH, subfolder='vae').to(device=DEVICE, dtype=torch.float32)

pil_convert = transforms.ToPILImage()
tensor_convert = transforms.ToTensor()

@torch.inference_mode()
def from_pil_to_latent(pil_img):
    pil_img = pil_img.resize((512, 512))
    img = tensor_convert(pil_img)
    img = img.unsqueeze(0).to(device=DEVICE, dtype=DTYPE)
    return encode(img)

@torch.inference_mode()
def from_latent_to_pil(latents):
    img = decode(latents)
    ret = [pil_convert(i) for i in img]
    if len(ret) == 1:
        return ret[0]
    else:
        return ret

@torch.inference_mode()
def get_condition(text):
    with torch.no_grad():
        input_ids = tokenizer(text, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        input_ids = input_ids.to(DEVICE)
        text_embedding = text_encoder(input_ids)[0]
    return {
        "encoder_hidden_states": text_embedding,
    }

@torch.inference_mode()
def encode(input_img): # Autoencoder takes [-1, 1] as input
    if len(input_img.shape)<4:
        input_img = input_img.unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode(input_img * 2 - 1) 
    return (vae.config.scaling_factor * latent.latent_dist.sample()).to(dtype=DTYPE)

@torch.inference_mode()
def decode(latents):
    latents = (1 / vae.config.scaling_factor) * latents
    MAX_CHUNK_SIZE = 2 # on 3090, we only have ~24GB of memory
    
    all_image = []
    for i in range(0, len(latents), MAX_CHUNK_SIZE):
        chunk = latents[i:min(i+MAX_CHUNK_SIZE, len(latents))]
        image = vae.decode(chunk.to(torch.float32)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().to(latents.dtype)
        all_image.append(image)
    image = torch.cat(all_image)
    image = image.detach().to(latents.dtype)
    return image

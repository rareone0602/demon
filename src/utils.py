import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as transforms
from tqdm import tqdm, trange

DEVICE = torch.device("cuda")
FILE_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
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



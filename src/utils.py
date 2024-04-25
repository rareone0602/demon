import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
import torchvision.transforms as transforms
from tqdm import tqdm, trange
from config import FILE_PATH, DTYPE, DEVICE

tokenizers = [
    CLIPTokenizer.from_pretrained(FILE_PATH, subfolder="tokenizer"),
    CLIPTokenizer.from_pretrained(FILE_PATH, subfolder="tokenizer_2")
]
text_encoders = [
    CLIPTextModel.from_pretrained(FILE_PATH, subfolder="text_encoder").to(device=DEVICE, dtype=DTYPE),
    CLIPTextModelWithProjection.from_pretrained(FILE_PATH, subfolder="text_encoder_2").to(device=DEVICE, dtype=DTYPE)
]
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
    prompts = [text, text]
    prompt_embeds_list = []
    for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):     
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(DEVICE), output_hidden_states=True)

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    prompt_embeds = prompt_embeds.to(dtype=text_encoders[1].dtype, device=DEVICE)
    
    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)


    add_text_embeds = pooled_prompt_embeds # tensor([[ 0.3979,  0.3870, -1.7441,  ..., -0.3667, -2.1816,  0.1017]], device='cuda:0', dtype=torch.float16)

    add_time_ids = torch.tensor(
        [[1024., 1024.,    0.,    0., 1024., 1024.]], 
        dtype=torch.float16).to(DEVICE).repeat(len(text), 1)
    # The shape is square for experimentation purposes
    
    return {
        "encoder_hidden_states": prompt_embeds, 
        "added_cond_kwargs": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
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
    MAX_CHUNK_SIZE = 1
    
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

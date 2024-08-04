# Reproduce the sampling in Diffusion-DPO 
import os
# Third-party library imports
import fire
import torch
import json
from transformers import AutoModel, AutoProcessor
from datasets import load_dataset
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from tqdm import trange
# Local application/library specific imports
from generate_abstract import DemonGenerater
import numpy as np

# 352 query or 2700s
pickscore_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
pickscore_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to('cuda')

def rewards(pils, text):
    image_inputs = pickscore_processor(
        images=pils,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to('cuda')
    text_inputs = pickscore_processor(
        text=text,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to('cuda')
    with torch.inference_mode():
        image_embs = pickscore_model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = pickscore_model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # score
        scores = (text_embs @ image_embs.T)[0]
        return scores.cpu().numpy().tolist()

def dpo_generate():
    # load pipeline
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    # unet_id = "mhdang/dpo-sdxl-text2image-v1"
    # unet = UNet2DConditionModel.from_pretrained(unet_id, subfolder="unet", torch_dtype=torch.float16)
    # pipe.unet = unet
    pipe = pipe.to("cuda")
        
    text_column = load_dataset("yuvalkirstain/pickapic_v2_no_images", split="test")
    text_column = text_column.shuffle(seed=42)["caption"]
    
    best_score_sum = 0
    all_prompts = list(set(text_column))[:80]

    best_scores = []
    for prompt in all_prompts:
        best_score = 0
        bar = trange(352) # Same number to compare with our method
        for _ in bar: 
            image = pipe(prompt, guidance_scale=5).images[0].resize((512,512))
            image.save('text.png')
            score = rewards([image], prompt)[0]
            if best_score < score:
                best_score = score
            bar.set_description(f"Score: {score:.8f}, Best: {best_score:.8f}")
        best_scores.append(best_score)
        print(f"Best score median: {np.median(best_scores)}")

if __name__ == '__main__':
    fire.Fire(dpo_generate)

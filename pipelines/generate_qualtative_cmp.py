from PIL import Image
from api import add_noise, get_init_latent, from_pil_to_latent, from_latent_to_pil, demon_sampling, sdeint, odeint
from datetime import datetime
from transformers import AutoProcessor, AutoModel
from helpers import AestheticScorerOld
import torch

aesthetic_scorer = AestheticScorerOld().to('cuda')

pickscore_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
pickscore_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to('cuda').half()

@torch.inference_mode()
def pickscore_reward(x):
    pil = from_latent_to_pil(x)
    inputs = pickscore_processor(images=pil, text="", return_tensors="pt", padding=True).to('cuda').half()
    outputs = pickscore_model(**inputs)
    return outputs.logits_per_image.item()

def reward(x):
    pil = from_latent_to_pil(x)
    aesthetic_score = aesthetic_scorer(pil).item()
    return aesthetic_score

if __name__ == '__main__':
    modified_img = Image.open('qual/dog/doodl.png').convert('RGB') # 5.21
    raw_img = Image.open('qual/dog/raw.png').convert('RGB')
    print(aesthetic_scorer([raw_img, modified_img]))

    latent = from_pil_to_latent(raw_img)
    aes_sum = 0
    start_t = 0.9
    latent_t = add_noise(latent, start_t) # 0.1 in DDPM is about 0.34 in Karras SDE
    latent_0 = demon_sampling(
        latent_t,
        pickscore_reward,
        {},
        beta=.5,
        tau=0.001,
        action_num=8,
        sample_step=32,
        weighting="spin",
        start_t=start_t,
        log_dir='qual/dog'
    )
    pil = from_latent_to_pil(latent_0)
    pil.save('qual/dog/pickscore.png')
    aes_score = aesthetic_scorer([pil]).item()
    print(aes_score)
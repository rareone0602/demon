# Standard library imports
import json
import os
from datetime import datetime

# Third-party library imports
import fire
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoProcessor

# Local application/library specific imports
from api import demon_sampling, get_init_latent, from_latent_to_pil, from_pil_to_latent, odeint
import ImageReward as RM

pickscore_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
pickscore_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to('cuda')

imageReward_model = RM.load("ImageReward-v1.0")


# The std of rm_reward and pickscore_reward is about 5 times larger than aesthetic_reward, so we divide them by 5
@torch.inference_mode()
def rm_reward(pil):
    return imageReward_model.score(prompt, [pil]) 

@torch.inference_mode()
def pickscore_reward(pil):
    inputs = pickscore_processor(images=pil, text=prompt, return_tensors="pt", padding=True).to('cuda')
    return pickscore_model(**inputs).logits_per_image.item() 

@torch.inference_mode()
def qr_similarity(pil):
    x_tensor = pil_to_tensor(pil)
    return (x_tensor * qrcode_tensor).mean().item()

def generate_pyplot(log_txt, out_img_file):
    scores = []
    std_devs = []
    ts = []
    with open(log_txt, "r") as f:
        for line in f.readlines():
            score, std_dev, t = map(float, line.split())
            scores.append(score)
            std_devs.append(std_dev)
            ts.append(t)
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.errorbar(ts, scores, yerr=std_devs, fmt='-o', capsize=5, capthick=1, ecolor='red', markeredgecolor = "black", color='blue')
    plt.title('Aesthetic Score vs Noise Level')
    plt.xlabel('t')
    plt.ylabel('Aesthetic Score')
    plt.gca().invert_xaxis()  # To display larger sigmas on the left
    plt.grid(True)
    plt.savefig(out_img_file)
    # close the plot to avoid memory leak
    plt.close()

def pil_to_tensor(pil):
    pil = pil.resize((512, 512)).convert('RGB')
    return ToTensor()(pil).unsqueeze(0).to('cuda') * 2 - 1

def qr_generate(
    beta=1,
    tau=0.05,
    action_num=16,
    sample_step=64,
    cfg=2,
    qr_similarity_weight=50,
    weighting="spin",
    text=None,
    seed=None,
    qrcode='assets/qrcode/rickroll.png',
    experiment_directory="experiments/qr_generate",
):
    # convert qrcode to tensor with [-1, 1] range
    
    
    
    
    global prompt
    global qrcode_tensor
    
    qrcode_tensor = pil_to_tensor(Image.open(qrcode))
    prompt = text

    if seed is None:
        # use unix epoch
        seed = int(datetime.now().timestamp())

    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(experiment_directory, datetime_str)
    os.makedirs(os.path.join(log_dir, 'trajectory'), exist_ok=False)

    def reward(x):
        pil = from_latent_to_pil(x)
        pil.save(f'{log_dir}/trajectory/{datetime.now().timestamp()}.png')
        rm = rm_reward(pil)
        pick = pickscore_reward(pil)
        qr = qr_similarity(pil)
        # std 0.1694455 , 0.16087078, 0.00368896
        return rm + pick + qr * qr_similarity_weight
        
    config = {
        "beta": beta,
        "tau": tau,
        "action_num": action_num,
        "sample_step": sample_step,
        "cfg": cfg,
        "qr_similarity_weight": qr_similarity_weight,
        "weighting": weighting,
        "prompt": prompt,
        "seed": seed,
        "log_dir": log_dir,
    }

        
    with open(f'{log_dir}/config.json', 'w') as f:
        json.dump(config, f)

    torch.manual_seed(seed)
    latent = get_init_latent()
    
    if prompt is None:
        prompts = {}
    else:
        prompts = {prompt: cfg}
    
    latent = demon_sampling(
        latent,
        reward,
        prompts,
        beta,
        tau,
        action_num,
        sample_step,
        weighting,
        log_dir=log_dir
    )
    pil = from_latent_to_pil(latent)
    pil.save(f'{log_dir}/{prompt}.png')
    generate_pyplot(f"{log_dir}/expected_energy.txt", f"{log_dir}/expected_energy.png")

    # {experiment_directory}/{datetime_str}.png


if __name__ == '__main__':
    fire.Fire(qr_generate)
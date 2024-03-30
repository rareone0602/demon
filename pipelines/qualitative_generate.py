# Standard library imports
import json
import os
from datetime import datetime

# Third-party library imports
import fire
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoProcessor

# Local application/library specific imports
from api import demon_sampling, get_init_latent, from_latent_to_pil, odeint
import ImageReward as RM
from reward_models.AestheticScorer import AestheticScorer


aesthetic_scorer = AestheticScorer().to('cuda')

pickscore_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
pickscore_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to('cuda')

imageReward_model = RM.load("ImageReward-v1.0")


# The std of rm_reward and pickscore_reward is about 5 times larger than aesthetic_reward, so we divide them by 5
@torch.inference_mode()
def rm_reward(pil):
    return imageReward_model.score(prompt, [pil]) / 5

@torch.inference_mode()
def pickscore_reward(pil):
    inputs = pickscore_processor(images=pil, text=prompt, return_tensors="pt", padding=True).to('cuda')
    return pickscore_model(**inputs).logits_per_image.item() / 5

@torch.inference_mode()
def aesthetic_reward(pil):
    aesthetic_score = aesthetic_scorer(pil).item()
    return aesthetic_score

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


def qualitative_generate(
    beta=.5,
    tau=0.1,
    action_num=16,
    sample_step=64,
    cfg=2,
    weighting="spin",
    text=None,
    seed=None,
    aesthetic=False,
    imagereward=False,
    pickscore=False,
    experiment_directory="experiments/qualitative_generate",
):
    def reward(x):
        pil = from_latent_to_pil(x)
        total = 0
        if aesthetic:
            total += aesthetic_reward(pil)
        if imagereward:
            total += rm_reward(pil)
        if pickscore:
            total += pickscore_reward(pil)
        return total
    
    global prompt
    prompt = text

    if seed is None:
        # use unix epoch
        seed = int(datetime.now().timestamp())

    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(experiment_directory, datetime_str)
    os.makedirs(log_dir, exist_ok=False)

    config = {
        "beta": beta,
        "tau": tau,
        "action_num": action_num,
        "sample_step": sample_step,
        "weighting": weighting,
        "prompt": prompt,
        "seed": seed,
        "aesthetic": aesthetic,
        "imagereward": imagereward,
        "pickscore": pickscore,
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
    
    if not any([aesthetic, imagereward, pickscore]):
        latent = odeint(latent, prompts, sample_step)
    else:
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
    fire.Fire(qualitative_generate)

# Experiments:
# CUDA_VISIBLE_DEVICES=9 python3 pipelines/qualitative_generate.py --text="Symmetry Product render poster vivid colors divine proportion owl glowing fog intricate elegant highly detailed" --seed=42 --imagereward
# CUDA_VISIBLE_DEVICES=8 python3 pipelines/qualitative_generate.py --text="Symmetry Product render poster vivid colors divine proportion owl glowing fog intricate elegant highly detailed" --seed=42 --aesthetic
# CUDA_VISIBLE_DEVICES=7 python3 pipelines/qualitative_generate.py --text="Symmetry Product render poster vivid colors divine proportion owl glowing fog intricate elegant highly detailed" --seed=42 --pickscore
# CUDA_VISIBLE_DEVICES=6 python3 pipelines/qualitative_generate.py --text="Symmetry Product render poster vivid colors divine proportion owl glowing fog intricate elegant highly detailed" --seed=42 --imagereward --pickscore
# CUDA_VISIBLE_DEVICES=5 python3 pipelines/qualitative_generate.py --text="Symmetry Product render poster vivid colors divine proportion owl glowing fog intricate elegant highly detailed" --seed=42 --aesthetic --imagereward --pickscore
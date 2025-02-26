# Standard Library Imports
import os
import json
from datetime import datetime

# Third-party Imports
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import fire
import numpy as np

# Local Application/Library Specific Imports
from api import add_noise, get_init_latent, best_of_n_sampling
from utils import from_latent_to_pil
from reward_models.AestheticScorer import AestheticScorer
from config import DTYPE, FILE_PATH

aesthetic_scorer = AestheticScorer()

def rewards(xs):
    """
    Calculate the aesthetic score of an image.
    """
    pils = from_latent_to_pil(xs)
    """
    os.makedirs(f'tmp/trajectory', exist_ok=True)
    nowtime = int(datetime.now().timestamp() * 1e6)
    for i, pil in enumerate(pils):
        pil.save(f'tmp/trajectory/{nowtime}_{i}.png')
    """
    return aesthetic_scorer(pils).cpu().numpy().tolist()

def read_animals(file_path):
    """
    Read a file containing a list of animals.
    """
    with open(file_path, 'r') as f:
        animals = f.read().splitlines()
    return animals

def generate_pyplot(log_txt, out_img_file):
    """
    Generate a plot of aesthetic scores vs noise level.
    """
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
    plt.errorbar(ts, scores, yerr=std_devs, fmt='-o', capsize=5, capthick=1, ecolor='red', markeredgecolor="black", color='blue')
    plt.title('Aesthetic Score vs Noise Level')
    plt.xlabel('t')
    plt.ylabel('Aesthetic Score')
    plt.gca().invert_xaxis()  # To display larger sigmas on the left
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.grid(True)
    plt.savefig(out_img_file)
    plt.close()

def aesthetic_animal_eval(
    best_of_n_step=50,
    beta=.5,
    tau='adaptive',
    action_num=16,
    weighting="spin",
    sample_step=64,
    timesteps="karras",
    max_ode_steps=20,
    ode_after=0.11,
    cfg=2,
    seed=42,
    experiment_directory="experiments/aesthetic_animal_eval",
):
    """
    Evaluate the aesthetic score of animals using latent space optimization.
    """
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(experiment_directory, datetime_str)
    os.makedirs(log_dir, exist_ok=False)

    config = {
        "beta": beta,
        "tau": tau,
        "action_num": action_num,
        "weighting": weighting,
        "sample_step": sample_step,
        "timesteps": timesteps,
        "max_ode_steps": max_ode_steps,
        "ode_after": ode_after,
        "cfg": cfg,
        "seed": seed,
        "log_dir": log_dir,
        "dtype": str(DTYPE),
        "file_path": FILE_PATH,
    }

    with open(f'{log_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=4)

    torch.manual_seed(seed)
    
    animals = read_animals('assets/very_simple_animal.txt')
    
    scores = []
    start_time = datetime.now()
    for prompt in tqdm(animals):
        prompts = {
            "prompts": [prompt],
            "cfgs": [cfg]
        }
        os.mkdir(os.path.join(log_dir, prompt))
        latent = best_of_n_sampling(
            best_of_n_step,
            rewards,
            prompts,
            beta,
            log_dir=os.path.join(log_dir, prompt),
        )
        pil = from_latent_to_pil(latent)
        pil.save(os.path.join(log_dir, prompt, f"out.png"))
        
        scores.append(aesthetic_scorer(pil).item())
    
    config["time"] =  (datetime.now() - start_time).total_seconds() / len(animals)
    scores = np.array(scores)
    # Find the time per instance
    config["score"] = np.mean(scores).item()
    config["score_std"] = np.std(scores).item()

    with open(f'{log_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=4)


if __name__ == '__main__':
    fire.Fire(aesthetic_animal_eval)

# CUDA_VISIBLE_DEVICES=9
# python3 pipelines/aesthetic_animal_eval.py \
# --beta 0.1 --action_num 16 --sample_step 64 --experiment_directory "experiments/rebuttal/aesthetic_animal_eval"
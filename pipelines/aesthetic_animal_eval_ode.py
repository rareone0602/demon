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
from api import add_noise, get_init_latent, odeint
from utils import from_latent_to_pil
from reward_models.AestheticScorer import AestheticScorer
from config import DTYPE, FILE_PATH

aesthetic_scorer = AestheticScorer()

def read_animals(file_path):
    """
    Read a file containing a list of animals.
    """
    with open(file_path, 'r') as f:
        animals = f.read().splitlines()
    return animals

def aesthetic_animal_eval(
    sample_step=64,
    timesteps="karras",
    cfg=2,
    seed=42,
    experiment_directory="experiments/ode_only",
):
    """
    Evaluate the aesthetic score of animals using latent space optimization.
    """
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(experiment_directory, datetime_str)
    os.makedirs(log_dir, exist_ok=False)

    config = {
        "sample_step": sample_step,
        "timesteps": timesteps,
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
        latent = odeint(
            get_init_latent(),
            prompts,
            sample_step,
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

# python3 pipelines/aesthetic_animal_eval_ode.py

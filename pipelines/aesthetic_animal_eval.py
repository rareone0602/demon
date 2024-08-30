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
from api import add_noise, get_init_latent, demon_sampling
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
            score, std_dev, t, _ = map(float, line.split())
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

def write_summary(log_dir, action_num):
    animals = read_animals('assets/very_simple_animal.txt')
    scores_across_animal = []
    for prompt in animals:
        data_file = os.path.join(log_dir, prompt, 'expected_energy.txt')
        second_passed = []
        scores_across_animal.append([])
        with open(data_file, 'r') as f:
            for line in f.readlines():
                score, _, _, time = map(float, line.split())
                second_passed.append(time)
                scores_across_animal[-1].append(score)
    scores_across_animal_mean = np.mean(scores_across_animal, axis=0)
    scores_across_animal_std = np.std(scores_across_animal, axis=0)
    second_passed = np.array(second_passed)    

    with open(os.path.join(log_dir, 'expected_energy.txt'), 'w') as f:
        for mean, std, t, i in zip(scores_across_animal_mean, scores_across_animal_std, second_passed, range(0, 1000, action_num)):
            f.write(f"{mean} {std} {t} {i}\n")
        

def aesthetic_animal_eval(
    beta=.5,
    tau='adaptive',
    action_num=16,
    weighting="spin",
    sample_step=64,
    r_of_c="baseline",
    c_steps=20,
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
        "r_of_c": r_of_c,
        "c_steps": c_steps,
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
        latent = demon_sampling(
            get_init_latent(),
            rewards,
            prompts,
            beta,
            tau,
            action_num,
            weighting,
            sample_step,
            r_of_c,
            c_steps=c_steps,
            ode_after=ode_after,
            log_dir=os.path.join(log_dir, prompt),
        )
        pil = from_latent_to_pil(latent)
        pil.save(os.path.join(log_dir, prompt, f"out.png"))
        # generate_pyplot(os.path.join(log_dir, prompt, 'expected_energy.txt'), 
        #                 os.path.join(log_dir, prompt, "expected_energy.png"))
        
        scores.append(aesthetic_scorer(pil).item())
    
    config["time"] =  (datetime.now() - start_time).total_seconds() / len(animals)
    scores = np.array(scores)
    # Find the time per instance
    config["score"] = np.mean(scores).item()
    config["score_std"] = np.std(scores).item()

    with open(f'{log_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    write_summary(log_dir, action_num)

if __name__ == '__main__':
    fire.Fire(aesthetic_animal_eval)
    
# CUDA_VISIBLE_DEVICES=9
# python3 pipelines/aesthetic_animal_eval.py \
# --beta 0.1 --action_num 16 --sample_step 64 --experiment_directory "experiments/rebuttal/aesthetic_animal_eval"
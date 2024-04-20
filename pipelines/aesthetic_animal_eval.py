# Standard Library Imports
import os
import json
from datetime import datetime

# Third-party Imports
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import fire

# Local Application/Library Specific Imports
from api import add_noise, get_init_latent, demon_sampling
from utils import from_latent_to_pil
from reward_models.AestheticScorer import AestheticScorer

aesthetic_scorer = AestheticScorer()

def rewards(xs):
    """
    Calculate the aesthetic score of an image.
    """
    return aesthetic_scorer(from_latent_to_pil(xs)).cpu().numpy().tolist()

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
    plt.grid(True)
    plt.savefig(out_img_file)
    plt.close()


def aesthetic_animal_eval(
    beta=.5,
    tau=0.1,
    action_num=16,
    sample_step=64,
    weighting="spin",
    cfg=2,
    seed=42,
    max_ode_steps=18,
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
        "sample_step": sample_step,
        "weighting": weighting,
        "cfg": cfg,
        "seed": seed,
        "max_ode_steps": max_ode_steps,
        "log_dir": log_dir
    }
    with open(f'{log_dir}/config.json', 'w') as f:
        json.dump(config, f)

    torch.manual_seed(seed)
    
    score_sum = 0
    
    animals = read_animals('assets/common_animals.txt')
    
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
            sample_step,
            weighting,
            log_dir=os.path.join(log_dir, prompt),
            max_ode_steps=max_ode_steps,
        )
        pil = from_latent_to_pil(latent)
        pil.save(os.path.join(log_dir, prompt, f"out.png"))
        generate_pyplot(os.path.join(log_dir, prompt, 'expected_energy.txt'), 
                        os.path.join(log_dir, prompt, "expected_energy.png"))
        
        score_sum += aesthetic_scorer(pil).item()
    
    config["score"] = score_sum / len(animals)

    with open(f'{log_dir}/config.json', 'w') as f:
        json.dump(config, f)


if __name__ == '__main__':
    fire.Fire(aesthetic_animal_eval)

# python3 pipelines/aesthetic_animal_eval.py --beta 0.5 --tau 0.1 --action_num 8 --sample_step 30 --weighting spin

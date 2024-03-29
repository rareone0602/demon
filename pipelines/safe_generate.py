import torch
import fire
import json
import os
from api import add_noise, get_init_latent, from_latent_to_pil, demon_sampling
from datetime import datetime
from transformers import AutoModelForImageClassification, ViTImageProcessor
import matplotlib.pyplot as plt

model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
processor = ViTImageProcessor.from_pretrained('Falconsai/nsfw_image_detection')

def reward(x):
    pil = from_latent_to_pil(x)
    with torch.no_grad():
        inputs = processor(images=pil, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
    return -logits[0, 1].item()


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
    plt.title('Negative NSFW Logit vs Noise Level')
    plt.xlabel('t')
    plt.ylabel('Negative NSFW Logit')
    plt.gca().invert_xaxis()  # To display larger sigmas on the left
    plt.grid(True)
    plt.savefig(out_img_file)


def safe_generate(
    beta=.5,
    tau=0.1,
    action_num=8,
    sample_step=30,
    weighting="spin",
    prompt=None,
    cfg=2,
    seed=42,
    experiment_directory="experiments/safe",
):
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
        "cfg": cfg,
        "seed": seed,
        "log_dir": log_dir
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


if __name__ == '__main__':
    fire.Fire(safe_generate)
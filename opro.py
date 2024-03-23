import json
from api import add_noise, get_init_latent, from_latent_to_pil, demon_sampling

from helpers import AestheticScorer, opro_gemini, opro_gpt
from tqdm import tqdm
import torch

import copy

aesthetic_scorer = AestheticScorer()
def read_jsonl_to_list(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_object = json.loads(line)
            data_list.append(json_object)
    return data_list

def add_json_to_jsonl(new_json, file_path):
    with open(file_path, 'a', encoding='utf-8') as file:
        json.dump(new_json, file)
        file.write('\n')
        
def read_animals(file_path):
    with open(file_path, 'r') as f:
        animals = f.read().splitlines()
        # Due to time constraints, we use the first 10 animals
        animals = animals[:10]
    return animals

def scoring_hyperparameters(hyperparameters):
    def reward(x):
        return aesthetic_scorer(from_latent_to_pil(x)).item()
    score_sum = 0
    bar = tqdm(zip(animals, fixed_latents))
    for animal, latent in bar:
        latent = demon_sampling(
            latent,
            reward, 
            {animal: 5, **hyperparameters["prompts"]},
            hyperparameters["beta"],
            hyperparameters["tau"],
            hyperparameters["action_num"], 
            hyperparameters["sample_step"], 
            hyperparameters["weighting"]
        )
        pil = from_latent_to_pil(latent)
        current_score = aesthetic_scorer(pil).item()
        score_sum += current_score
        pil.save(f'test_img/{animal}.png')
        bar.set_description(f"Current score for {animal}: {current_score}")
    score_mean = score_sum / len(animals)
    return score_mean - (len(hyperparameters["prompts"]) + 1) * hyperparameters["action_num"] * hyperparameters["sample_step"] / 1000

hyperparameters_list = read_jsonl_to_list('hyperparameters_list.jsonl')
current_hyperparameters = max(hyperparameters_list, key=lambda x: x["score"])["hyperparameters"]
animals = read_animals('assets/common_animals.txt')[:10]
torch.manual_seed(42)
fixed_latents = [get_init_latent() for _ in range(len(animals))]

for key, options in [('sample_step', [32, 64, 128])]:
    for option in options:
        current_hyperparameters[key] = option
        for weighting in ["spin", "boltzmann"]:
            current_hyperparameters["weighting"] = weighting
            reported_score = scoring_hyperparameters(current_hyperparameters)
            add_json_to_jsonl({
                "hyperparameters": copy.deepcopy(current_hyperparameters.copy()),
                "score": reported_score
            }, 'hyperparameters_list.jsonl')
            hyperparameters_list = read_jsonl_to_list('hyperparameters_list.jsonl')
            
    current_hyperparameters = max(hyperparameters_list, key=lambda x: x["score"])["hyperparameters"]

pos_prompt = 'Mystical landscapes and enchanted forests, Celestial skies with whimsical creatures, Vibrant and dynamic lighting, High clarity, Attention to HD textures, Rich color saturation without oversaturation, Natural representation free from digital glitches or unrealistic elements'
neg_prompt = 'low-quality, blurry imagesm, watermarks, logos, banners, cropped pictures, jpeg artifacts, Distortion, and unwanted geometrical mutations, low resolution'
current_hyperparameters['prompts'] = {pos_prompt: 2, neg_prompt: -2}

for pos_cfg, neg_cfg in [(1, -1), (4, -4), (4, -2), (8, -2), (16, -2)]:
    current_hyperparameters['prompts'] = {pos_prompt: pos_cfg, neg_prompt: neg_cfg}
    reported_score = scoring_hyperparameters(current_hyperparameters)
    add_json_to_jsonl({
        "hyperparameters": copy.deepcopy(current_hyperparameters.copy()),
        "score": reported_score
    }, 'hyperparameters_list.jsonl')
    hyperparameters_list = read_jsonl_to_list('hyperparameters_list.jsonl')

current_hyperparameters = max(hyperparameters_list, key=lambda x: x["score"])["hyperparameters"]


while True:
    current_hyperparameters = opro_gpt(hyperparameters_list, model="gpt-4-0125-preview")
    reported_score = scoring_hyperparameters(current_hyperparameters)
    add_json_to_jsonl({
        "hyperparameters": copy.deepcopy(current_hyperparameters.copy()),
        "score": reported_score
    }, 'hyperparameters_list.jsonl')
    hyperparameters_list = read_jsonl_to_list('hyperparameters_list.jsonl')
    
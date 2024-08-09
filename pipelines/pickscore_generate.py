# Reproduce the sampling in Diffusion-DPO 
import os
# Third-party library imports
import fire
import torch
import json
from transformers import AutoModel, AutoProcessor
from datasets import load_dataset

# Local application/library specific imports
from generate_abstract import DemonGenerater


pickscore_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
pickscore_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to('cuda')
class QualitativeGenerater(DemonGenerater):
    def rewards(self, pils):
        image_inputs = pickscore_processor(
            images=pils,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to('cuda')
    
        text_inputs = pickscore_processor(
            text=prompt,
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

def pickscore_generate(
    beta=.05,
    tau='adaptive',
    action_num=16,
    weighting="spin",
    sample_step=32,
    timesteps="karras",
    max_ode_steps=20,
    ode_after=0.11,
    cfg=2,
    seed=None,
    save_pils=True,
    experiment_directory="experiments/pickscore_generate",
):
    generated_prompts = set()
    previous_dirs = os.listdir(experiment_directory)
    for previous_dir in previous_dirs:
        config = json.load(open(f'{experiment_directory}/{previous_dir}/config.json'))
        generated_prompts.add(config['prompt'])

    global prompt
    generator = QualitativeGenerater(
            beta=beta,
            tau=tau,
            action_num=action_num,
            weighting=weighting,
            sample_step=sample_step,
            timesteps=timesteps,
            max_ode_steps=max_ode_steps,
            ode_after=ode_after,
            cfg=cfg,
            seed=seed,
            save_pils=save_pils,
            experiment_directory=experiment_directory
        )
    text_column = load_dataset("yuvalkirstain/pickapic_v2_no_images", split="test")
    text_column = text_column.shuffle(seed=seed)["caption"]
    for prompt in list(set(text_column)):
        if prompt in generated_prompts:
            continue
        generator.generate(prompt=prompt)

if __name__ == '__main__':
    fire.Fire(pickscore_generate)

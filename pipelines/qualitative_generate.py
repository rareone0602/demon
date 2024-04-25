# Third-party library imports
import fire
import torch
import json
from transformers import AutoModel, AutoProcessor

# Local application/library specific imports
import ImageReward as RM
from reward_models.AestheticScorer import AestheticScorer
from generate_abstract import DemonGenerater
import hpsv2


aesthetic_scorer = AestheticScorer().to('cuda')

pickscore_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
pickscore_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to('cuda')

imageReward_model = RM.load("ImageReward-v1.0")

@torch.inference_mode()
def hpsv2_reward(pil):
    return hpsv2.score(pil, prompt, hps_version="v2.1")[0] * 40 # We scale the reward by 40 to match the scale of other rewards

# The std of rm_reward and pickscore_reward is about 5 times larger than aesthetic_reward, so we divide them by 5
@torch.inference_mode()
def rm_reward(pil):
    return imageReward_model.score(prompt, [pil]) 

@torch.inference_mode()
def pickscore_reward(pil):
    inputs = pickscore_processor(images=pil, text=prompt, return_tensors="pt", padding=True).to('cuda')
    return pickscore_model(**inputs).logits_per_image.item() 

@torch.inference_mode()
def aesthetic_reward(pil):
    aesthetic_score = aesthetic_scorer(pil).item()
    return aesthetic_score

def qualitative_generate(
    beta=.5,
    tau='adaptive',
    action_num=16,
    weighting="spin",
    sample_step=64,
    timesteps="karras",
    max_ode_steps=20,
    ode_after=0.11,
    text=None,
    cfg=2,
    seed=None,
    aesthetic=False,
    imagereward=False,
    pickscore=False,
    hpsv2=False,
    experiment_directory="experiments/qualitative_generate",
):
    global prompt
    prompt = text
    def reward(pil):
        total = 0
        if aesthetic:
            total += aesthetic_reward(pil)
        if imagereward:
            total += rm_reward(pil)
        if pickscore:
            total += pickscore_reward(pil)
        if hpsv2:
            total += hpsv2_reward(pil)
        return total
    
    class QualitativeGenerater(DemonGenerater):
        def rewards(self, pils):
            return [reward(pil) for pil in pils]
        
        def generate(self, prompt):
            super().generate(prompt, ode=not any([aesthetic, imagereward, pickscore, hpsv2]))

            with open(f'{self.log_dir}/config.json', 'r') as f:
                config = json.load(f)
                config['aesthetic'] = aesthetic
                config['imagereward'] = imagereward
                config['pickscore'] = pickscore
                config['hpsv2'] = hpsv2
                
            with open(f'{self.log_dir}/config.json', 'w') as f:
                json.dump(config, f, indent=4)

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
        experiment_directory=experiment_directory
    )

    generator.generate(prompt=text)

if __name__ == '__main__':
    fire.Fire(qualitative_generate)

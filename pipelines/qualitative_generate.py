# Third-party library imports
import fire
import torch
from transformers import AutoModel, AutoProcessor

# Local application/library specific imports
import ImageReward as RM
from reward_models.AestheticScorer import AestheticScorer
from generate_abstract import DemonGenerater

aesthetic_scorer = AestheticScorer().to('cuda')

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
def aesthetic_reward(pil):
    aesthetic_score = aesthetic_scorer(pil).item()
    return aesthetic_score

def qualitative_generate(
    beta=.5,
    tau=0.05,
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
        return total
    
    class QualitativeGenerater(DemonGenerater):
        def rewards(self, pils):
            return [reward(pil) for pil in pils]
    
    generator = QualitativeGenerater(
        beta=beta,
        tau=tau,
        action_num=action_num,
        sample_step=sample_step,
        cfg=cfg,
        weighting=weighting,
        seed=seed,
        experiment_directory=experiment_directory,
    )

    generator.generate(prompt=text)

if __name__ == '__main__':
    fire.Fire(qualitative_generate)

# Experiments:
# CUDA_VISIBLE_DEVICES=9 python3 pipelines/qualitative_generate.py --text="Symmetry Product render poster vivid colors divine proportion owl glowing fog intricate elegant highly detailed" --seed=42 --imagereward
# CUDA_VISIBLE_DEVICES=8 python3 pipelines/qualitative_generate.py --text="Symmetry Product render poster vivid colors divine proportion owl glowing fog intricate elegant highly detailed" --seed=42 --aesthetic
# CUDA_VISIBLE_DEVICES=7 python3 pipelines/qualitative_generate.py --text="Symmetry Product render poster vivid colors divine proportion owl glowing fog intricate elegant highly detailed" --seed=42 --pickscore
# CUDA_VISIBLE_DEVICES=6 python3 pipelines/qualitative_generate.py --text="Symmetry Product render poster vivid colors divine proportion owl glowing fog intricate elegant highly detailed" --seed=42 --imagereward --pickscore
# CUDA_VISIBLE_DEVICES=5 python3 pipelines/qualitative_generate.py --text="Symmetry Product render poster vivid colors divine proportion owl glowing fog intricate elegant highly detailed" --seed=42 --aesthetic --imagereward --pickscore
# Third-party library imports
import fire
from image_grid import create_image_grid
from generate_abstract import DemonGenerater
from utils import from_latent_to_pil
import os
from datetime import datetime

class ChooseGenerator(DemonGenerater):
    def rewards_latent(self, latents):
        pils = from_latent_to_pil(latents)
        rs = self.rewards(pils)
        nowtime = int(datetime.now().timestamp() * 1e6)
        if self.save_pils:
            os.makedirs(f'{self.log_dir}/trajectory', exist_ok=True)
            for i, (pil, r) in enumerate(zip(pils, rs)):
                pil.save(f'{self.log_dir}/trajectory/{nowtime}_{i}_{r}.png')
        
        return rs
        
    def rewards(self, pils):
        return create_image_grid(pils)

def choose_generate(
    beta=.5,
    tau='adaptive',
    action_num=16,
    weighting="spin",
    sample_step=128,
    c_steps=22,
    r_of_c="baseline",
    ode_after=0.11,
    text=None,
    cfg=1,
    seed=None,
    save_pils=True,
    experiment_directory="experiments/choose_generate",
):
    
        
    generator = ChooseGenerator(
        beta=beta,
        tau=tau,
        action_num=action_num,
        weighting=weighting,
        sample_step=sample_step,
        c_steps=c_steps,
        r_of_c=r_of_c,
        ode_after=ode_after,
        cfg=cfg,
        seed=seed,
        save_pils=save_pils,
        experiment_directory=experiment_directory
    )

    generator.generate(prompt=text)

if __name__ == '__main__':
    fire.Fire(choose_generate)

# python pipelines/choose_generate.py --text "A boulder in elevator"
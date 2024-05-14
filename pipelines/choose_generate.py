# Third-party library imports
import fire
from image_grid import create_image_grid
from generate_abstract import DemonGenerater

class ChooseGenerator(DemonGenerater):
        def rewards(self, pils):
            return create_image_grid(pils)

def choose_generate(
    beta=.1,
    tau='adaptive',
    action_num=16,
    weighting="spin",
    sample_step=128,
    timesteps="karras",
    max_ode_steps=22,
    ode_after=0.11,
    text=None,
    cfg=2,
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
        timesteps=timesteps,
        max_ode_steps=max_ode_steps,
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
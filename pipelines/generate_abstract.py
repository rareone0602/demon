# Standard library imports
import json
import os
from datetime import datetime

# Third-party library imports
import torch
import matplotlib.pyplot as plt
import numpy as np

# Local application/library specific imports
from api import demon_sampling, get_init_latent, odeint
from utils import from_latent_to_pil
from abc import ABC, abstractmethod


class DemonGenerater(ABC):

    def __init__(
            self, 
            beta=.5, 
            tau='adaptive',
            action_num=16, 
            weighting="spin",
            sample_step=64,
            timesteps="karras",
            max_ode_steps=20,
            ode_after=0.11,
            cfg=2,
            seed=None,
            save_pils=False,
            ylabel="Energy",
            experiment_directory="experiments/generate",
        ):
        self.beta = beta
        self.tau = tau
        self.action_num = action_num
        self.weighting = weighting
        self.sample_step = sample_step
        self.timesteps = timesteps
        self.max_ode_steps = max_ode_steps
        self.ode_after = ode_after
        self.cfg = cfg
        self.save_pils = save_pils
        self.ylabel = ylabel
        self.experiment_directory = experiment_directory
        
        if seed is None:
            self.seed = int(datetime.now().timestamp())
        else:
            self.seed = seed

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def rewards_latent(self, latents):
        pils = from_latent_to_pil(latents)

        if self.save_pils:
            os.makedirs(f'{self.log_dir}/trajectory', exist_ok=True)
            nowtime = int(datetime.now().timestamp() * 1e6)
            for i, pil in enumerate(pils):
                pil.save(f'{self.log_dir}/trajectory/{nowtime}_{i}.png')
        
        return self.rewards(pils)

    @abstractmethod
    def rewards(self, pils):
        pass

    def generate_pyplot(self, log_txt, out_img_file):
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
        plt.errorbar(ts, scores, yerr=std_devs, fmt='-o', capsize=5, capthick=1, ecolor='red', markeredgecolor = "black", color='blue')
        plt.title(f'{self.ylabel} vs Noise Level t')
        plt.xlabel('t')
        plt.ylabel(self.ylabel)
        plt.gca().invert_xaxis()  # To display larger sigmas on the left
        plt.xscale('log')  # Set x-axis to logarithmic scale
        plt.grid(True)
        plt.savefig(out_img_file)
        plt.close()

    def generate(self, prompt, ode=False):
        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(self.experiment_directory, datetime_str)
        os.makedirs(self.log_dir, exist_ok=False)

        self.config = {
            "beta": self.beta,
            "tau": self.tau,
            "action_num": self.action_num,
            "weighting": self.weighting,
            "sample_step": self.sample_step,
            "timesteps": self.timesteps,
            "max_ode_steps": self.max_ode_steps,
            "ode_after": self.ode_after,            
            "prompt": prompt,
            "cfg": self.cfg,
            "seed": self.seed,
            "log_dir": self.log_dir,
        }

        with open(f'{self.log_dir}/config.json', 'w') as f:
            json.dump(self.config, f, indent=4)
        
        latent = get_init_latent()
        
        from_latent_to_pil(
            odeint(
                latent,
                {
                    "prompts": [prompt if prompt is not None else ""],
                    "cfgs": [self.cfg]
                },
                self.sample_step
            )
        ).save(f'{self.log_dir}/init.png')

        if not ode:
            latent = demon_sampling(
                latent,
                self.rewards_latent,
                {
                    "prompts": [prompt if prompt is not None else ""],
                    "cfgs": [self.cfg]
                },
                self.beta,
                self.tau,
                self.action_num,
                self.weighting,
                self.sample_step,
                self.timesteps,
                max_ode_steps=self.max_ode_steps,
                ode_after=self.ode_after,
                log_dir=self.log_dir,
            )
        
            from_latent_to_pil(latent).save(f'{self.log_dir}/out.png')
            self.generate_pyplot(f"{self.log_dir}/expected_energy.txt", f"{self.log_dir}/expected_energy.png")

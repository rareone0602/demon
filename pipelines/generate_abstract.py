# Standard library imports
import json
import os
from datetime import datetime

# Third-party library imports
import torch
import matplotlib.pyplot as plt

# Local application/library specific imports
from api import demon_sampling, get_init_latent, from_latent_to_pil, odeint
from abc import ABC, abstractmethod


class DemonGenerater(ABC):

    def __init__(self, 
                 beta=0.5, 
                 tau=0.05,
                 action_num=16, 
                 sample_step=64,
                 cfg=2,
                 weighting="spin",
                 seed=None,
                 save_pils=False,
                 ylabel="Energy",
                 experiment_directory="experiments/generate",
                 ode_after_sigma=0
                 ):
        self.beta = beta
        self.tau = tau
        self.action_num = action_num
        self.sample_step = sample_step
        self.cfg = cfg
        self.weighting = weighting
        self.seed = seed
        self.save_pils = save_pils
        self.ylabel = ylabel
        self.experiment_directory = experiment_directory
        self.ode_after_sigma = ode_after_sigma

        if seed is None:
            seed = int(datetime.now().timestamp())
        
        torch.manual_seed(seed)

    def rewards_latent(self, latents):
        pils = [from_latent_to_pil(latent.unsqueeze(0)) for latent in latents]

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
                score, std_dev, t = map(float, line.split())
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
            "sample_step": self.sample_step,
            "weighting": self.weighting,
            "prompt": prompt,
            "seed": self.seed,
            "log_dir": self.log_dir,
        }

        with open(f'{self.log_dir}/config.json', 'w') as f:
            json.dump(self.config, f)
        
        if ode:
            latent = odeint(
                get_init_latent(),
                {} if prompt is None else {prompt: self.cfg},
                self.sample_step,
            )
        else:
            latent = demon_sampling(
                get_init_latent(),
                self.rewards_latent,
                {} if prompt is None else {prompt: self.cfg},
                self.beta,
                self.tau,
                self.action_num,
                self.sample_step,
                self.weighting,
                log_dir=self.log_dir,
                ode_after_sigma=self.ode_after_sigma
            )
        
        from_latent_to_pil(latent).save(f'{self.log_dir}/out.png')
    
        if not ode:
            self.generate_pyplot(f"{self.log_dir}/expected_energy.txt", f"{self.log_dir}/expected_energy.png")

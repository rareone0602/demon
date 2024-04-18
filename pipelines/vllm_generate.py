# Third-party library imports
import fire
from transformers import AutoModel, AutoProcessor
from generate_abstract import DemonGenerater
from llm import ask_gemini, ask_gpt
import os

with open('assets/scenarios.txt', 'r') as f:
    scenarios = f.readlines()


class VLLMGenerater(DemonGenerater):

    def __init__(self, senario, prompt, model='gemini', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.senario = senario
        self.prompt = prompt
        self.choices = []
        if model == 'gemini':
            self.ask = ask_gemini
        elif model == 'gpt':
            self.ask = ask_gpt
        else:
            raise ValueError("Unknown model.")

    def rewards(self, pils):
        if len(pils) % 2 == 1:
            raise ValueError("The number of action_num should be even.")
        choice = []
        for i in range(0, len(pils), 2):
            choice += self.ask(self.senario, self.prompt, pils[i:i+2])
        self.choices.append(choice)
        return choice

def vllm_generate(
    beta=.5,
    tau=0.001,
    action_num=16,
    sample_step=128,
    cfg=2,
    weighting="spin",
    text="A mysterious, glowing object discovered in an unexpected place, sparking curiosity and wonder. The setting changes based on the viewer's background, transforming the object's significance and the surrounding environment to match the realms of education, history, literature, design, science, and imagination.",
    seed=None,
    model='gemini',
    experiment_directory="experiments/vllm_as_demon",
):  
    for scenario in scenarios:
        exp_dir = os.path.join(experiment_directory, scenario.split(' ')[3])
        os.makedirs(exp_dir, exist_ok=True)
        generator = VLLMGenerater(
            senario=scenario,
            prompt=text,
            model=model,
            beta=beta,
            tau=tau,
            action_num=action_num,
            sample_step=sample_step,
            cfg=cfg,
            weighting=weighting,
            seed=seed,
            save_pils=True,
            ylabel="Decision",
            experiment_directory=exp_dir,
            ode_after_sigma=0.12 # To save the number of unnecessary calls to the ODE solver
        )
        generator.generate(prompt=text)

        with open(f'{exp_dir}/choices.txt', 'a') as f:
            f.write(str(generator.choices))

if __name__ == '__main__':
    fire.Fire(vllm_generate)
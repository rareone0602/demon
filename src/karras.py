import torch
import torch.nn as nn
import numpy as np

from diffusers import KDPM2DiscreteScheduler, UNet2DConditionModel, DPMSolverMultistepScheduler
from scipy.interpolate import InterpolatedUnivariateSpline

def to_float(t):
    """Convert supported types to float.
    
    Args:
        t: Variable to convert, can be of type torch.Tensor, np.ndarray, or float.
        
    Returns:
        A float representation of the input.
    """
    if type(t) == torch.Tensor or type(t) == np.ndarray:
        return t.item()
    try:
        return float(t)
    except:
        raise ValueError(f"Unsupported type {type(t)}")


class SigmaScoreModel(nn.Module):
    """Computes the Sigma-Score for the unscaled value.

    Args:
        unet: The U-Net model for computing noise.
        s: Scaling values, shape (M,)
    """
    def __init__(self, unet, scheduler):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self.M = len(scheduler.num_train_timesteps)
        sigmas = np.array(((1 - scheduler.alphas_cumprod) / scheduler.alphas_cumprod) ** 0.5)
        self.scheduler.log_sigmas = np.log(sigmas)
    
    def sigma_to_t(self, sigma):
        """Convert sigma to t."""
        return self.scheduler._sigma_to_t(sigma, self.scheduler.log_sigmas)
    
    def compute_noise_prediction(self, sigma, x, prompt_embedding):
        """Compute noise prediction given time, input, and prompt embedding."""
        if prompt_embedding.shape[0] == 1:
            prompt_embedding = prompt_embedding.expand(x.shape[0], -1, -1)
        return self.unet(
            self.scheduler.scale_model_input(x),
            self.sigma_to_t(sigma) * (self.M - 1), 
            encoder_hidden_states=prompt_embedding, 
            return_dict=False
            )[0]

    def get_noise(self, sigma, x, prompt_embedding):
        """Fetch noise prediction for a given time and input."""
        noise_prediction = self.compute_noise_prediction(sigma, x, prompt_embedding)
        return noise_prediction

    def forward(self, sigma, x, prompt_embedding):
        return -self.get_noise(sigma, x, prompt_embedding)
    
class LatentSDEModel(nn.Module):
    """
    Stochastic Differential Equation model
    """
    def __init__(self, beta='anderson', const=None, path='runwayml/stable-diffusion-v1-5'):
        super().__init__()
        unet = UNet2DConditionModel.from_pretrained(path, subfolder='unet').to('cuda')
        scheduler = KDPM2DiscreteScheduler.from_pretrained(path, subfolder='scheduler')
        self.sigma_score = SigmaScoreModel(unet, scheduler)
        self.change_noise(beta=beta, const=const)
    
    def ode_mode(self):
        self.prev_beta = self.beta
        self.change_noise(const=0)
    
    def ode_mode_revert(self):
        self.beta = self.prev_beta
    
    def change_noise(self, beta='anderson', const=None):
        if const is not None:
            self.beta = lambda t: const**2 / t**2 / 2
        elif beta == 'anderson':
            self.beta = lambda t: 1 / t
        elif isinstance(beta, (int, float)):
            self.beta = lambda t: beta
        else:
            assert callable(beta), "Expected beta to be a lambda function"
            self.beta = beta

    def get_timesteps(self, N, sigma_max=1., sigma_min=0.005):
        RHO = 7
        B, A = sigma_max**(1/RHO), (sigma_min**(1/RHO) - sigma_max**(1/RHO)) / N
        # Karras: rho=3 nearly equalizes the truncation error
        return torch.FloatTensor([(A * (i + 1) + B)**RHO for i in range(N)]).to(self.sigma_score.unet.device)

    def prepare_initial_latents(self, batch_size=1, height=512, width=512):
        VAE_SCALE_FACTOR = 8
        NUM_CHANNEL_LATENTS = 4
        shape = (batch_size, NUM_CHANNEL_LATENTS, height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR)
        return torch.randn(shape, device=self.sigma_score.unet.device, dtype=self.sigma_score.unet.dtype) * self.init_noise_sigma
        
    def forward(self, sigma, latent, text_embedding):
        return self.f(sigma, latent, text_embedding), self.g(sigma, latent, text_embedding)
        return f, g
    
    def f(self, t, y, text_embedding):
        sigma_score_val =  self.sigma_score(t, y, text_embedding)
        if torch.is_tensor(t) and t.ndim > 0:
            mul = torch.stack([
                torch.tensor(-self.sigma_score.karras.sigma_dot(a) - self.karras.beta(a) * self.karras.sigma(a), device=t.device) for a in t
            ]).view(-1, 1, 1, 1)
            return mul * sigma_score_val
        else:
            return (-self.sigma_score.karras.sigma_dot(t) - self.karras.beta(t) * self.karras.sigma(t)) * sigma_score_val
    
    def g(self, t, y, text_embedding):
        if torch.is_tensor(t) and t.ndim > 0:
            return torch.stack([
                torch.tensor((2 * self.karras.beta(a))**0.5 * self.karras.sigma(a), device=t.device) for a in t
            ])
        else:
            return (2 * self.karras.beta(t))**0.5 * self.karras.sigma(t)

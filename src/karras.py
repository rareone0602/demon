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


class KarrasArgs:
    """Handles interpolations related to Karras et al.'s paper.
    
    Args:
        s_values: Scaling values w.r.t to discrete schedule, shape (M,)
    """
    def __init__(self, s_values):
        self.initialize_interpolators(s_values)

    def initialize_interpolators(self, s_values):
        """Initialize the interpolators for a, sigma, and s based on s_values."""
        t_values = np.linspace(1e-5, 1, len(s_values))
        a_values = np.log(s_values)
        sigma_values = np.sqrt(1 - s_values**2) / s_values

        self.create_interpolator('a', t_values, a_values)
        self.create_interpolator('sigma', t_values, sigma_values)
        self.create_interpolator('s', t_values, s_values)
        self.create_interpolator('sigma_inv', sigma_values, t_values)

    def create_interpolator(self, name, x, y):
        """Create and store an interpolator and its derivative.
        
        Args:
            name: Name of the interpolator ('a', 'sigma', or 's').
            x, y: Data points for the interpolation.
        """
        setattr(self, f"{name}_interpolator", InterpolatedUnivariateSpline(x, y, k=3))
        setattr(self, f"{name}_dot_interpolator", getattr(self, f"{name}_interpolator").derivative())

    def get_interpolated_value(self, interpolator, t):
        """Fetch interpolated value for a given t using the interpolator."""
        return interpolator(to_float(t)).item()

    def __getattr__(self, name):
        interpolator = getattr(self, f"{name}_interpolator", None)
        return lambda t: self.get_interpolated_value(interpolator, t)

class SigmaScoreModel(nn.Module):
    """Computes the Sigma-Score for the unscaled value.

    Args:
        unet: The U-Net model for computing noise.
        s: Scaling values, shape (M,)
    """
    def __init__(self, unet, s):
        super().__init__()
        self.unet = unet
        self.karras = KarrasArgs(s)
        self.M = len(s)

    def compute_noise_prediction(self, t, x, prompt_embedding):
        """Compute noise prediction given time, input, and prompt embedding."""
        if prompt_embedding.shape[0] == 1:
            prompt_embedding = prompt_embedding.expand(x.shape[0], -1, -1)
        return self.unet(x, t * (self.M - 1), encoder_hidden_states=prompt_embedding, return_dict=False)[0]

    def get_noise(self, t, x, prompt_embedding):
        """Fetch noise prediction for a given time and input."""
        if torch.is_tensor(t) and t.ndim > 0:
            x_input = torch.stack([x[i] * self.karras.s(t[i]) for i in range(t.shape[0])])
        else:
            x_input = x * self.karras.s(t)
        noise_prediction = self.compute_noise_prediction(t, x_input, prompt_embedding)
        return noise_prediction

    def forward(self, t, x, prompt_embedding):
        return -self.get_noise(t, x, prompt_embedding)
    
class LatentSDEModel(nn.Module):
    """
    Stochastic Differential Equation model
    """
    def __init__(self, beta='anderson', const=None, path='runwayml/stable-diffusion-v1-5'):
        super().__init__()
        unet = UNet2DConditionModel.from_pretrained(path, subfolder='unet').to('cuda')
        scheduler = KDPM2DiscreteScheduler.from_pretrained(path, subfolder='scheduler')
        
        self.init_noise_sigma = scheduler.init_noise_sigma
        self.scheduler = scheduler
        self.sigma_score = SigmaScoreModel(unet, np.sqrt(scheduler.alphas_cumprod.cpu().numpy()))
        self.karras = self.sigma_score.karras
        self.change_noise(beta=beta, const=const)
    
        
    def ode_mode(self):
        self.prev_beta = self.karras.beta
        self.change_noise(const=0)
    
    def ode_mode_revert(self):
        self.karras.beta = self.prev_beta
    
    def change_noise(self, beta='anderson', const=None):
        if const is not None:
            self.karras.beta = lambda t: const**2 / self.karras.sigma(t)**2 / 2
        elif beta == 'anderson':
            self.karras.beta = lambda t: self.karras.sigma_dot(t) / self.karras.sigma(t)
        elif isinstance(beta, (int, float)):
            self.karras.beta = lambda t: beta
        else:
            assert callable(beta), "Expected beta to be a lambda function"
            self.karras.beta = beta

    def get_timesteps(self, N, start_time=1., end_time=0.):
        sigma_min, sigma_max = self.karras.sigma(end_time), self.karras.sigma(start_time)
        RHO = 7
        B, A = sigma_max**(1/RHO), (sigma_min**(1/RHO) - sigma_max**(1/RHO)) / N
        # Karras: rho=3 nearly equalizes the truncation error
        return torch.FloatTensor([self.karras.sigma_inv((A * (i + 1) + B)**RHO) for i in range(N)]).to(self.sigma_score.unet.device)

    def prepare_initial_latents(self, batch_size=1, height=512, width=512):
        VAE_SCALE_FACTOR = 8
        NUM_CHANNEL_LATENTS = 4
        shape = (batch_size, NUM_CHANNEL_LATENTS, height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR)
        return torch.randn(shape, device=self.sigma_score.unet.device, dtype=self.sigma_score.unet.dtype) * self.init_noise_sigma
    
    def scale(self, latent, t):
        return latent * self.karras.s(t)

    def unscale(self, scaled_latent, t):
        return scaled_latent / self.karras.s(t)
        
    def forward(self, t, latent, text_embedding):
        f, g = self.f(t, latent, text_embedding), self.g(t, latent, text_embedding)
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

import torch
import torch.nn as nn
import numpy as np

from diffusers import KDPM2DiscreteScheduler, UNet2DConditionModel
from scipy.interpolate import InterpolatedUnivariateSpline
from config import FILE_PATH, DTYPE, DEVICE, IMAGE_DIMENSION

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
        self.M = scheduler.config.num_train_timesteps
        sigmas = np.array(((1 - scheduler.alphas_cumprod) / scheduler.alphas_cumprod) ** 0.5)
        
        # Note: the scheduler use linear interpolation, but we use cubic spline for smoother interpolation.
        
        lower_bound = 0.9858077
        
        # Found by binary search
        # 1e-10 => postive but very close to 0
        # sigmas[0] => ?
        # sigmas[-1] => self.M - 1
        
        self._sigma_to_t = InterpolatedUnivariateSpline(
            sigmas,
            np.linspace(lower_bound, self.M - 1, self.M), 
            k=3
        )

    def sigma_to_t(self, sigma):
        """Convert sigma to t."""
        return torch.from_numpy(
            self._sigma_to_t(sigma.cpu().numpy())
        ).to(sigma.device)
    
    def sigma_to_scale(self, sigma):
        """Convert sigma to scale."""
        return 1 / ((sigma**2 + 1) ** 0.5)
    
        
    def _compute_noise_prediction(self, sigma, x, conds_kwargs):
        """Compute noise prediction given time, input, and prompt embedding."""
        discrete_t = self.sigma_to_t(sigma) # of range(M)
        return self.unet(
            self.sigma_to_scale(sigma) * x,
            discrete_t,
            return_dict=False,
            **conds_kwargs
            )[0]

    def _get_noise(self, sigma, x, conds):
        """Fetch noise prediction for a given time and input."""
        noise_prediction = self._compute_noise_prediction(sigma, x, conds)
        return noise_prediction

    def forward(self, sigma, x, conds):
        return -self._get_noise(sigma, x, conds)
    
class LatentSDEModel(nn.Module):
    """
    Stochastic Differential Equation model
    """
    def __init__(self, beta='anderson', const=None, path=FILE_PATH):
        super().__init__()
        unet = UNet2DConditionModel.from_pretrained(path, subfolder='unet').to(dtype=DTYPE, device=DEVICE)
        scheduler = KDPM2DiscreteScheduler.from_pretrained(path, subfolder='scheduler')
        self.sigma_score = SigmaScoreModel(unet, scheduler)
        self.change_noise(beta=beta, const=const)
    
    def ode_mode(self):
        self.prev_beta = self.beta
        self.change_noise(beta=0)
    
    def ode_mode_revert(self):
        self.beta = self.prev_beta
    
    def change_noise(self, beta='anderson', const=None):
        if const is not None:
            self.beta = lambda sigma: const**2 / sigma**2 / 2
        elif beta == 'anderson':
            self.beta = lambda sigma: 1 / sigma
        elif isinstance(beta, (int, float)):
            self.beta = lambda sigma: beta
        else:
            assert callable(beta), "Expected beta to be a lambda function"
            self.beta = beta

    def get_log_timesteps(self, T, sigma_max=14.6488, sigma_min=1e-4):
        base = 2
        A, B = base**(sigma_min), base**(sigma_max)
        return torch.Tensor([np.log(A + ((T - 1 - i) / (T - 1)) * (B - A)) / np.log(base) for i in range(T)]).to(dtype=DTYPE, device=DEVICE)
    
    def get_linear_timesteps(self, T, sigma_max=14.6488, sigma_min=1e-4):
        return torch.linspace(sigma_max, sigma_min, T).to(dtype=DTYPE, device=DEVICE)

    def get_karras_timesteps(self, T, sigma_max=14.6488, sigma_min=1e-4):
        RHO = 7
        A, B = sigma_min**(1/RHO), sigma_max**(1/RHO)
        return torch.Tensor([(A + ((T - 1 - i) / (T - 1)) * (B - A))**RHO for i in range(T)]).to(dtype=DTYPE, device=DEVICE)

    def prepare_initial_latents(self, batch_size=1, height=IMAGE_DIMENSION, width=IMAGE_DIMENSION):
        VAE_SCALE_FACTOR = 8
        NUM_CHANNEL_LATENTS = 4
        shape = (batch_size, NUM_CHANNEL_LATENTS, height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR)
        return torch.randn(shape, device=DEVICE, dtype=DTYPE) * self.sigma_score.scheduler.init_noise_sigma
        
    def forward(self, sigma, latent, conds):
        return self.f(sigma, latent, conds), self.g(sigma)
    
    def f(self, sigma, x, conds):
        sigma_score_val =  self.sigma_score(sigma, x, conds)
        return (-self.beta(sigma) * sigma - 1) * sigma_score_val
    
    def g(self, sigma):
        return (2 * self.beta(sigma))**0.5 * sigma
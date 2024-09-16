import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime

from karras import LatentSDEModel, LatentConsistencyModel
from utils import get_condition
from config import DEVICE, DTYPE
import copy

latent_sde = LatentSDEModel(beta=0)
# latent_consistency = LatentConsistencyModel()

class OdeModeContextManager:
    def __enter__(self):
        latent_sde.ode_mode()

    def __exit__(self, exc_type, exc_val, exc_tb):
        latent_sde.ode_mode_revert()
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

def duplicate_condition(conds, n):
    return {
        "encoder_hidden_states": conds["encoder_hidden_states"].repeat(n, 1, 1), 
        "added_cond_kwargs": {
            "text_embeds": conds["added_cond_kwargs"]["text_embeds"].repeat(n, 1), 
            "time_ids": conds["added_cond_kwargs"]["time_ids"].repeat(n, 1),
            "timestep_cond": conds["timestep_cond"].repeat(n, 1) if conds["timestep_cond"] is not None else None,
        },
    }

def _get_f_g(t, x, prompts):
    conds = prompts['conditions']
    cfgs = prompts['cfgs']
    N, C = x.shape[0], len(cfgs) + 1
    if N > 1:
        conds = duplicate_condition(conds, x.shape[0])
    fs, g = latent_sde(
        t, 
        x.repeat_interleave(C, dim=0), 
        conds
    )

    all_f = []
    for j in range(N):
        f = fs[(j + 1) * C - 1] + \
            sum((fs[j * C + i] - fs[(j + 1) * C - 1]) * cfg for i, cfg in enumerate(cfgs))
        all_f.append(f)
    
    return torch.stack(all_f), g

@torch.inference_mode()
def get_f_g(t, x, prompts):
    MAX_CHUNK_SIZE = 8
    # 16 if you only need aesthetic score on 3090
    N = x.shape[0]
    all_fs = []
    for i in range(0, N, MAX_CHUNK_SIZE):
        chunk = x[i:min(i+MAX_CHUNK_SIZE, N)]
        fs, g = _get_f_g(t, chunk, prompts) # g is assumed to be the same for all elements in the chunk
        all_fs.append(fs)
    
    return torch.cat(all_fs), g

@torch.inference_mode()
def sde_step(x, t, prev_t, prompts, z):
    assert x.shape[0] == 1
    # Note: z may have more batch dimensions than x
    dt = t - prev_t
    rand_term = z * torch.sqrt(torch.abs(dt))
    f1, g1 = get_f_g(prev_t, x, prompts)
    x_pred = x + f1 * dt + g1 * rand_term # Expand x to match z's shape
    f2, g2 = get_f_g(t, x_pred, prompts)
    x = x + 0.5 * (f1 + f2) * dt + 0.5 * (g1 + g2) * rand_term
    return x

@torch.inference_mode()
def ode_step(x, t, prev_t, prompts):
    dt = t - prev_t
    f1, _ = get_f_g(prev_t, x, prompts)
    x_pred = x + f1 * dt
    f2, _ = get_f_g(t, x_pred, prompts)
    x = x + 0.5 * (f1 + f2) * dt
    return x

def best_stepnum(t, sigma_max=14.6488, sigma_min=2e-3, max_ode_step=18, RHO=7):
    A, B, C = sigma_min**(1/RHO), sigma_max**(1/RHO), t**(1/RHO)
    ratio = (C - A) / (B - A)
    return max(np.ceil(max_ode_step * ratio).astype(int), 0)
    
@OdeModeContextManager()
@torch.inference_mode()
def odeint_rest(x, start_t, ts, prompts, max_ode_steps=18):
    steps = best_stepnum(start_t.item(), max_ode_step=max_ode_steps) + 2
    ts = latent_sde.get_karras_timesteps(steps, start_t, sigma_min=ts[-1])
    prev_t, ts = ts[0], ts[1:]
    for t in ts:
        x = ode_step(x, t, prev_t, prompts)
        prev_t = t
    return x

@OdeModeContextManager()
@torch.inference_mode()
def odeint(x, text_cfg_dict, sample_step, start_t=14.648, end_t=2e-3):
    ts = latent_sde.get_karras_timesteps(
        T=sample_step,
        sigma_max=start_t,
        sigma_min=end_t
    )
    text_cfg_dict = copy.deepcopy(text_cfg_dict)
    text_cfg_dict['prompts'].append('')
    prompts = {
        'conditions': get_condition(text_cfg_dict['prompts']),
        'cfgs': text_cfg_dict['cfgs'],
    }
    prev_t = ts[0]
    progress_bar = ts[1:]
    for t in progress_bar:
        x = ode_step(x, t, prev_t, prompts)
        prev_t = t
    return x

@torch.inference_mode()
def consistency_sampling(x, conds, sample_step=2, start_t=14.648, end_t=2e-3):
    # Note: z may have more batch dimensions than x
    ts = latent_sde.get_karras_timesteps(
        T=sample_step,
        sigma_max=start_t,
        sigma_min=end_t
    )
    N = x.shape[0]
    while len(ts) > 1:
        t, ts = ts[0], ts[1:]
        x = latent_consistency(t, x, conds)
        if len(ts) > 1:
            z = torch.randn_like(x[0:1]) # to reduce the uncertainty caused by independent noise
            x = x + ts[0] * z.repeat(N, 1, 1, 1)
    
    return x

@torch.inference_mode()
def sdeint(x, text_cfg_dict, beta, sample_step, start_t=14.648, end_t=2e-3):
    latent_sde.change_noise(beta=beta)
    ts = latent_sde.get_karras_timesteps(
        T=sample_step,
        sigma_max=start_t,
        sigma_min=end_t
    )
    text_cfg_dict['prompts'].append('')
    prompts = {
        'conditions': get_condition(text_cfg_dict['prompts']),
        'cfgs': text_cfg_dict['cfgs'],
    }
    prev_t = ts[0]
    for t in ts[1:]:
        x = sde_step(x, t, prev_t, prompts, torch.randn_like(x))
        prev_t = t
    return x

@torch.inference_mode()
def best_of_n_sampling(
        best_of_n_step,
        energy_fn, 
        text_cfg_dict, 
        beta, 
        sample_step=20,
        max_ode_steps=20,
        start_t=14.648,
        end_t=2e-3,
        log_dir=None):
    
    start_time = datetime.now()
    latent_sde.change_noise(beta=beta)
    
    ts = latent_sde.get_karras_timesteps(
        T=sample_step,
        sigma_max=start_t,
        sigma_min=end_t
    )
    text_cfg_dict['prompts'].append('')
    prompts = {
        'conditions': get_condition(text_cfg_dict['prompts']),
        'cfgs': text_cfg_dict['cfgs'],
    }
    prev_t, ts = ts[0], ts[1:]

    x = None
    score = -1
    
    for _ in range(best_of_n_step):
        cur_x = get_init_latent()
        x_0 = odeint_rest(cur_x, prev_t, ts, prompts, max_ode_steps=max_ode_steps)
        cur_score = energy_fn(x_0)[0]
        if cur_score > score:
            x, score = cur_x, cur_score
        passed_seconds = (datetime.now() - start_time).total_seconds()
        with open(f"{log_dir}/expected_energy.txt", "a") as f:
                f.write(f"{score} 0 NA {passed_seconds}\n")
    x_0 = odeint_rest(x, prev_t, ts, prompts, max_ode_steps=max_ode_steps)
    return x_0


@torch.inference_mode()
def demon_sampling(x, 
                   energy_fn, 
                   text_cfg_dict, 
                   beta, 
                   tau, 
                   action_num, 
                   weighting, 
                   sample_step,
                   r_of_c="baseline",
                   c_steps=20,
                   ode_after=0,
                   start_t=14.648,
                   end_t=2e-3,
                   log_dir=None):
    if r_of_c not in ["baseline", "consistency"]:
        raise ValueError(f"Unknown r_of_c: {r_of_c}")
    if x.shape[0] != 1:
        raise ValueError("Input x must have batch size of 1")

    latent_sde.change_noise(beta=beta)
    
    ts = latent_sde.get_karras_timesteps(
        T=sample_step,
        sigma_max=start_t,
        sigma_min=end_t
    )
    
    text_cfg_dict = copy.deepcopy(text_cfg_dict)
    text_cfg_dict['prompts'].append('')

    prompts = {
        'conditions': get_condition(text_cfg_dict['prompts']),
        'cfgs': text_cfg_dict['cfgs'],
    }

    if r_of_c == "consistency":
        cond = get_condition(text_cfg_dict['prompts'][:-1], time_cond=True)
        if action_num > 1:
            cond = duplicate_condition(cond, action_num)

    prev_t, ts = ts[0], ts[1:]

    start_time = datetime.now()
    while len(ts) > 0:
        if prev_t < ode_after:
            x = odeint_rest(x, prev_t, ts, prompts, max_ode_steps=18)
            break
        zs = torch.randn(action_num, *x.shape[1:], device=x.device, dtype=x.dtype)
        
        t, ts = ts[0], ts[1:]
        next_xs = sde_step(x, t, prev_t, prompts, zs)    
        
        if r_of_c == "consistency":
            candidates_0 = consistency_sampling(next_xs, cond, sample_step=c_steps+1, start_t=t)            
        elif r_of_c == "baseline":
            candidates_0 = odeint_rest(next_xs, t, ts, prompts, max_ode_steps=c_steps)
        
        values = torch.tensor(energy_fn(candidates_0)).to(device=x.device, dtype=x.dtype)
        
        passed_seconds = (datetime.now() - start_time).total_seconds()
        if log_dir is not None:
            with open(f"{log_dir}/expected_energy.txt", "a") as f:
                f.write(f"{values.mean().item()} {values.std().item()} {t.item()} {passed_seconds}\n")

        if tau == 'adaptive':
            tau = values.std().item()
        
        if weighting == "spin":
            values = values - values.mean()
            weights = torch.tanh(values / tau)
        elif weighting == "boltzmann":
            stabilized_values = values - torch.max(values)
            weights = F.softmax(stabilized_values / tau, dim=0)
        else:
            raise ValueError(f"Unknown weighting: {weighting}")
        if values.std().item() < 1e-8:
            weights = torch.ones_like(weights)
        z = F.normalize((zs * weights.view(-1, 1, 1, 1)).sum(dim=0, keepdim=True), dim=(0, 1, 2, 3)) # (1, C, H, W)
        z *= x.numel()**0.5
        x = sde_step(x, t, prev_t, prompts, z)
        prev_t = t
    return x

def add_noise(latent, t):
    z = torch.randn_like(latent)
    return latent + z * t

def get_init_latent():
    return latent_sde.prepare_initial_latents()
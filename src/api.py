import torch
import torch.nn.functional as F

from karras import LatentSDEModel
from utils import get_condition, from_latent_to_pil, from_pil_to_latent

latent_sde = LatentSDEModel(beta=0).to('cuda')

@torch.inference_mode()
def get_f_g(t, x, prompts):
    conds = prompts['conditions']
    cfgs = prompts['cfgs']
    fs, g = latent_sde(t, x.expand(len(cfgs) + 1, -1, -1, -1), conds)
    f = fs[-1:] + sum((fs[i] - fs[-1]).unsqueeze(0) * cfg for i, cfg in enumerate(cfgs))
    return f, g

@torch.inference_mode()
def sde_step(x, next_t, t, prompts, z):
    assert x.shape[0] == 1
    # Note: z may have more batch dimensions than x
    dt = next_t - t
    rand_term = z * torch.sqrt(torch.abs(dt))

    f1, g1 = get_f_g(t, x, prompts)
    x_pred = x + f1 * dt + g1 * rand_term # Expand x to match z's shape
    f2, g2 = get_f_g(next_t, x_pred, prompts)
    x = x + 0.5 * (f1 + f2) * dt + 0.5 * (g1 + g2) * rand_term
    return x

@torch.inference_mode()
def odeint_rest(x, start_t, ts, prompts):
    if len(ts) > 25:
        ts = latent_sde.get_timesteps(25, start_time=ts[0])
    prev_t = start_t
    for t in ts:
        dt = t - prev_t
        f1, _ = get_f_g(prev_t, x, prompts)
        x_pred = x + f1 * dt
        f2, _ = get_f_g(t, x_pred, prompts)
        x = x + 0.5 * (f1 + f2) * dt
        prev_t = t
    return x

@torch.inference_mode()
def odeint(x, text_cfg_dict, sample_step):
    ts = latent_sde.get_timesteps(
        T=sample_step, 
        sigma_max=latent_sde.sigma_score.scheduler.init_noise_sigma
    )
    text_cfg_dict['prompts'].append('')
    # convert text_weight_pair to 
    prompts = {
        'conditions': get_condition(text_cfg_dict['prompts']),
        'cfgs': text_cfg_dict['cfgs'],
    }
    prev_t = ts[0]
    for t in ts[1:]:
        dt = t - prev_t
        f1, _ = get_f_g(prev_t, x, prompts)
        x_pred = x + f1 * dt
        f2, _ = get_f_g(t, x_pred, prompts)
        x = x + 0.5 * (f1 + f2) * dt
        prev_t = t
    return x


@torch.inference_mode()
def sdeint(x, text_weight_pair, beta, sample_step, start_t=1., end_t=0.):
    latent_sde.change_noise(beta=beta)
    ts = latent_sde.get_timesteps(sample_step, start_t, end_t)
    prompts = [
        (get_embedding(prompt), weight) for prompt, weight in text_weight_pair.items()
    ]
    prev_t = start_t
    for t in ts:
        dt = t - prev_t
        z = torch.randn_like(x)
        rand_term = z * torch.sqrt(torch.abs(dt))
        f1, g1 = get_f_g(prev_t, x, prompts)
        x_pred = x + f1 * dt + g1 * rand_term
        f2, g2 = get_f_g(t, x_pred, prompts)
        x = x + 0.5 * (f1 + f2) * dt + 0.5 * (g1 + g2) * rand_term
        prev_t = t
    return x


@torch.inference_mode()
def demon_sampling(x, energy_fn, text_weight_pair, beta, tau, action_num, sample_step, weighting="spin", log_dir=None, start_t=1., end_t=0.):
    assert x.shape[0] == 1
    latent_sde.change_noise(beta=beta)
    ts = latent_sde.get_timesteps(sample_step, start_t, end_t)
    prompts = [
        (get_embedding(prompt), weight) for prompt, weight in text_weight_pair.items()
    ]
    prev_t = start_t
    while len(ts) > 0:
        t, ts = ts[0], ts[1:]
        zs = torch.randn(action_num, *x.shape[1:]).to(x.device)
        next_x = sde_step(x, t, prev_t, prompts, zs)
        latent_sde.ode_mode()
        candidate_0 = odeint_rest(next_x, t, ts, prompts)
        latent_sde.ode_mode_revert()

        values = torch.tensor(energy_fn(candidate_0))
        
        if log_dir is not None:
            # Append values.mean().item() and values.std().item() to {log_dir}/sample_hist.txt
            with open(f"{log_dir}/expected_energy.txt", "a") as f:
                f.write(f"{values.mean().item()} {values.std().item()} {latent_sde.karras.sigma(t.item())}\n")

        values = values - values.mean()
        
        if weighting == "spin":
            weights = torch.tanh(values / tau).to(x.device)
        elif weighting == "boltzmann":
            weights = F.softmax(values / tau, dim=0).to(x.device)
        else:
            raise ValueError(f"Unknown weighting: {weighting}")

        z = F.normalize((zs * weights.view(-1, 1, 1, 1)).sum(dim=0, keepdim=True), dim=(0, 1, 2, 3)) # (1, C, H, W)
        z *= x.numel()**0.5
        x = sde_step(x, t, prev_t, prompts, z)
        prev_t = t.item()
    return x



def add_noise(latent, t):
    z = torch.randn_like(latent)
    print(f"Sigma = {latent_sde.karras.sigma(t)}")
    return latent + z * latent_sde.karras.sigma(t)

def get_init_latent():
    return latent_sde.prepare_initial_latents()



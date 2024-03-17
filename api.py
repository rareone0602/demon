import torch
import torch.nn.functional as F

from karras import LatentSDEModel
from utils import get_embedding, sdeint, from_latent_to_pil, from_pil_to_latent

from tqdm import tqdm

latent_sde = LatentSDEModel(beta=0)
empty_text_embedding = get_embedding("")


@torch.inference_mode()
def get_f_g(t, x, prompts):
    f_emp, g_emp = latent_sde(t, x, empty_text_embedding)
    f1 = 0
    for text_embedding, cfg in prompts:
        f, _ = latent_sde(t, x, text_embedding)
        f1 += (f - f_emp) * cfg
    return f_emp + f1, g_emp

@torch.inference_mode()
def sde_step(x, next_t, t, prompts, z):
    assert x.shape[0] == 1
    dt = next_t - t
    rand_term = z * torch.sqrt(torch.abs(dt))

    f1, g1 = get_f_g(t, x, prompts)
    x_pred = x + f1 * dt + g1 * rand_term
    f2, g2 = get_f_g(next_t, x_pred, prompts)
    x = x + 0.5 * (f1 + f2) * dt + 0.5 * (g1 + g2) * rand_term
    return x


@torch.inference_mode()
def odeint(x, start_t, ts, prompts):
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
def demon_sampling(x, energy_fn, text_weight_pair, beta, tau, action_num, sample_step, weighting="tanh"):
    assert x.shape[0] == 1
    latent_sde.change_noise(beta=beta)
    ts = latent_sde.get_timesteps(sample_step, 1., 0.)
    prompts = [
        (get_embedding(prompt), weight) for prompt, weight in text_weight_pair.items()
    ]
    prev_t = 1
    while len(ts) > 0:
        t, ts = ts[0], ts[1:]
        zs = torch.randn(action_num, *x.shape[1:]).to(x.device)
        next_x = sde_step(x, t, prev_t, prompts, zs)
        latent_sde.ode_mode()
        candidate_0 = odeint(next_x, t, ts, prompts)
        latent_sde.ode_mode_revert()

        values = torch.tensor([energy_fn(candidate_0[i].unsqueeze(0)) for i in range(action_num)])
        values = values - values.mean()
        
        if weighting == "tanh":
            weights = torch.tanh(values / tau).to(x.device)
        elif weighting == "boltzmann":
            weights = F.softmax(values / tau, dim=0).to(x.device)

        z = F.normalize((zs * weights.view(-1, 1, 1, 1)).sum(dim=0, keepdim=True), dim=(0, 1, 2, 3)) # (1, C, H, W)
        z *= x.numel()**0.5
        x = sde_step(x, t, prev_t, prompts, z)
        prev_t = t.item()
    return x



def add_noise(latent, t):
    z = torch.randn_like(latent)
    return latent + z * latent_sde.karras.sigma(t)

def get_init_latent():
    return latent_sde.prepare_initial_latents()


if __name__ == "__main__":
    """
    prompts = [
        ("A shota boy's hand", 1),
        ("bad anatomy", -1)
    ]
    init_latent = get_init_latent()
    latents_0 = [
        latent_diffusion(init_latent, 1, 0, prompts, beta=0.1) for _ in range(4)
    ]

    imgs = [
        from_latent_to_pil(latent) for latent in latents_0
    ]

    print(get_description(["Which one best fits a drawing of a hand, or if the hand is not drawn incorrectly?", *imgs]))

    for i in range(4):
        imgs[i].save(f'test_img/{i}.png')
    """
    prompts = [
        ("A boy's hand", 1),
        ("bad anatomy", -1) # negative weight to discourage bad anatomy
    ]
    init_latent = get_init_latent()
    latents_0 = latent_diffusion(init_latent, 1, 0, prompts, beta=0.1)

    latents_half = add_noise(latents_0, 0.7)
    latent_0_edit = latent_diffusion(latents_half, 0.7, 0, prompts, beta=0)

    img = from_latent_to_pil(latent_0_edit)

    print(get_description(["Describe the image", img]))

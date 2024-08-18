import torch
DTYPE = torch.float16
FILE_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
C_FILE_PATH = "latent-consistency/lcm-sdxl"
DEVICE = torch.device("cuda")
IMAGE_DIMENSION = 1024
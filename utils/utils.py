import torch
from torchvision.transforms import Resize, InterpolationMode


def resize(input_tensor: torch.Tensor, target: torch.Tensor):
    return Resize(target.size()[-2:-1], interpolation=InterpolationMode.NEAREST).forward(input_tensor)

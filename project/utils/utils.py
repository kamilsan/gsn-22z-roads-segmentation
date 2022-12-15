import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def resize(input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return TF.resize(input_tensor, target.size()[-2:-1], interpolation=InterpolationMode.NEAREST)

from typing import Iterable

import torch


def z_clip(x: torch.Tensor, z_value: float, dims: Iterable[int] | None = (2, 3)):
    with torch.no_grad():
        std, mean = torch.std_mean(x, dim=dims, keepdim=True)
        min = mean - std * z_value
        max = mean + std * z_value
    x = torch.clip(x, min=min, max=max)
    return x

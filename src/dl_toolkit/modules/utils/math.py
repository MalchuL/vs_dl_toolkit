import math
from typing import Iterable

import numpy as np
import torch


def logit(value: Iterable[float] | float | torch.Tensor | np.ndarray) -> Iterable[float] | float | torch.Tensor | np.ndarray:
    """Logit function.

    Args:
        value (Iterable[float] | float | torch.Tensor | np.ndarray): Value to apply logit function.

    Returns:
        Iterable[float] | float | torch.Tensor | np.ndarray: Logit of the value.
    """
    if isinstance(value, torch.Tensor):
        return torch.logit(value)
    elif isinstance(value, np.ndarray):
        return np.log(value) - np.log(1 - value)
    elif isinstance(value, Iterable):
        return [logit(v) for v in value]
    else:
        return math.log(value) - math.log(1 - value)

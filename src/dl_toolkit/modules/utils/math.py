import math
from typing import Iterable

import numpy as np
import torch


def logit(value: Iterable[float] | float | torch.Tensor | np.ndarray):
    if isinstance(value, (list, tuple)):
        return [logit(v) for v in value]
    if isinstance(value, torch.Tensor):
        return torch.logit(value)
    elif isinstance(value, np.ndarray):
        return np.log(value) - np.log(1 - value)
    else:
        return math.log(value) - math.log(1 - value)

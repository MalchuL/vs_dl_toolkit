from enum import Enum, auto

import torch


class ReductionType(Enum):
    MEAN = auto()
    SUM = auto()
    NONE = auto()


def reduce_data(data: torch.Tensor, reduction: str | ReductionType | None) -> torch.Tensor:
    """Reduce data by reduction type.

    Args:
        data (torch.Tensor): Data to reduce.
        reduction (str | ReductionType | None): Reduction type.

    Returns:
        torch.Tensor: Reduced data.
    """
    if reduction in [ReductionType.MEAN, "mean"]:
        return data.mean()
    elif reduction in [ReductionType.SUM, "sum"]:
        return data.sum()
    elif reduction in [ReductionType.NONE, "none", None]:
        return data
    else:
        raise ValueError(f"Unknown reduction type: {reduction}")

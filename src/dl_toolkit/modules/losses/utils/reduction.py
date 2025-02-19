from enum import Enum, auto


class ReductionType(Enum):
    MEAN = auto()
    SUM = auto()
    NONE = auto()


def reduce_data(data, reduction: str | ReductionType | None):
    if reduction in [ReductionType.MEAN, 'mean']:
        return data.mean()
    elif reduction in [ReductionType.SUM, 'sum']:
        return data.sum()
    elif reduction in [ReductionType.NONE, 'none', None]:
        return data
    else:
        raise ValueError(f"Unknown reduction type: {reduction}")

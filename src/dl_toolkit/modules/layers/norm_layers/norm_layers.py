import functools
from typing import Type, Union, Callable

import torch.nn as nn

from .group_norm import GroupNorm, GroupNorm8


def get_norm_layer(norm_type: Union[str, Type[nn.Module]] = "instance") -> (
    Callable[[int], nn.Module]):
    """Returns a normalization layer class or constructor based on the input type.

    Supports common normalization layers like batch, instance, and group norm, or allows direct
    use of a custom `nn.Module` subclass. Configures default parameters for specific norm types.

    Args:
        norm_type (Union[str, Type[nn.Module]]): Specifies the type of normalization layer.
            - `"batch"`: `nn.BatchNorm2d` with `affine=True`.
            - `"instance"`: `nn.InstanceNorm2d` with `affine=False` and `track_running_stats=False`.
            - `"group"`: Returns the `GroupNorm` class.
            - `"group8"`: Returns the `GroupNorm8` class.
            - `"none"` or `None`: Returns `nn.Identity`.
            - A subclass of `nn.Module`: Returns the class directly.
            Defaults to `"instance"`.

    Returns:
        Union[functools.partial, Type[nn.Module]]: A configured normalization layer constructor
        (as a `functools.partial` object) or the class itself if a custom subclass is provided.

    Raises:
        NotImplementedError: If `norm_type` is an unsupported string.

    Example:
        >>> batch_norm_layer = get_norm_layer("batch")
        >>> layer = batch_norm_layer(64)  # Creates BatchNorm2d with 64 channels
        >>> isinstance(layer, nn.BatchNorm2d)
        True
    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "group":
        norm_layer = GroupNorm
    elif norm_type == "group8":
        norm_layer = GroupNorm8
    elif norm_type == "none" or norm_type is None:
        norm_layer = nn.Identity
    elif isinstance(norm_type, type) and issubclass(norm_type, nn.Module):
        return norm_type
    elif isinstance(norm_type, Callable):
        return norm_type
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer

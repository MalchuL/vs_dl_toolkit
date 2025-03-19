from typing import Mapping, Optional

import torch.nn as nn

from .loss_wrapper import LossWrapper


class MergingLossWrapper(nn.Module):
    """A module that combines multiple loss functions into a single weighted sum.

    This wrapper takes a dictionary of loss modules and an optional dictionary of weights.
    During the forward pass, it computes each loss, multiplies it by its corresponding weight,
    and returns the sum of all weighted losses.

    Args:
        losses (Mapping[str, nn.Module]): A dictionary mapping loss names to loss modules.
        weights (Optional[Mapping[str, float]]): A dictionary mapping loss names to weights.
            If None, all weights default to 1.0. Defaults to None.

    Example:
        >>> losses = {'mse': nn.MSELoss(), 'l1': nn.L1Loss()}
        >>> weights = {'mse': 0.5, 'l1': 0.5}
        >>> loss_wrapper = MergingLossWrapper(losses, weights)
        >>> output = loss_wrapper(predictions, targets)
    """

    def __init__(self, losses: Mapping[str, nn.Module],
                 weights: Optional[Mapping[str, float]] = None):
        """Initializes the MergingLossWrapper with the given losses and weights."""
        super().__init__()
        if weights is None:
            weights = {k: 1.0 for k in losses.keys()}

        self.losses = nn.ModuleList(
            [LossWrapper(loss, weight=weights[k]) for k, loss in losses.items()])

    def forward(self, *args, **kwargs):
        """Computes the weighted sum of all losses.

        Args:
            *args: Arguments passed to each loss module.
            **kwargs: Keyword arguments passed to each loss module.

        Returns:
            torch.Tensor: The total loss as a sum of weighted individual losses.
        """
        total_loss = 0.0
        for loss_wrapper in self.losses:
            total_loss += loss_wrapper(*args, **kwargs)
        return total_loss
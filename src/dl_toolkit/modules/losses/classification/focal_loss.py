import torch.nn as nn
import torchvision.ops as ops

from dl_toolkit.modules.toolkit_module import ToolkitModule
from dl_toolkit.utils.logging import logger


class FocalLoss(ToolkitModule):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "mean",
        *,
        warn_on_wrong_input: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.was_warning_input = not warn_on_wrong_input  # Make it true if we don't want to warn

    def forward(self, prediction, target):
        """
        Focal loss.
        Args:
            prediction (torch.Tensor): Prediction tensor in logits.
            target (torch.Tensor): Target tensor.

        Returns:

        """
        if not self.was_warning_input:
            if prediction.min() >= 0 and prediction.max() <= 1:
                logger.warning(
                    "Your focal loss prediction look like sigmoid output, "
                    "please check your input. You must pass logits to focal loss."
                )
            self.was_warning_input = True
        return ops.sigmoid_focal_loss(
            prediction, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction
        )

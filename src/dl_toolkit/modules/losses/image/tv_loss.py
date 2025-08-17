import random

import torch

from dl_toolkit.modules.toolkit_module import ToolkitModule


class TVLoss(ToolkitModule):
    def __init__(self, shift_size: int = 1, reduction: str = "mean"):
        """Total Variation loss.
        TV loss is a regularization term that penalizes the total variation of the image.
        It is used to prevent the model from overfitting to the input data.
        It is also used to improve the quality of the generated image.
        It is also used to improve the quality of the generated image.
        Args:
            shift_size (int, optional): Shift size for TV loss. Defaults to 1.
            reduction (str, optional): Reduction type. Defaults to "mean".
        """
        super().__init__()
        assert shift_size >= 1
        self.shift_size = shift_size
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Unknown reduction {reduction}")
        self.reduction = reduction

    def forward(self, x):
        shift = random.randint(1, self.shift_size)
        N, C, H, W = x.shape
        h_tv = torch.pow((x[:, :, shift:, :] - x[:, :, : H - shift, :]), 2)
        w_tv = torch.pow((x[:, :, :, shift:] - x[:, :, :, : W - shift]), 2)
        if self.reduction == "mean":
            return h_tv.mean() + w_tv.mean()
        elif self.reduction == "sum":
            return (h_tv.sum() + w_tv.sum()) / 2
        elif self.reduction == "none":
            return h_tv.flatten(1) + w_tv.flatten(1)
        else:
            raise ValueError(f"Unknown reduction {self.reduction}")

import torch
import torch.nn as nn

from dl_toolkit.modules.toolkit_module import ToolkitModule


class TClipLoss(ToolkitModule):
    def __init__(self, min_value: float = 0, max_value: float = 1, reduction: str = "mean"):
        super().__init__()
        self.min = min_value
        self.max = max_value
        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, tensor):
        mask = torch.clip(tensor, min=self.min, max=self.max)
        loss = self.loss(tensor, mask)
        return loss

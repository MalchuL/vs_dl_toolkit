import torch
import torch.nn as nn

from dl_toolkit.modules.losses.utils.reduction import reduce_data
from dl_toolkit.modules.toolkit_module import ToolkitModule


# Refer to https://arxiv.org/pdf/1806.05764.pdf
class CharbonnierLoss(ToolkitModule):
    """Charbonnier Loss (L1)"""

    VERSION = "1.0.1"

    def __init__(self, eps=1e-3, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self._eps_square = self.eps ** 2
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, x, y):
        loss = torch.sqrt(self.mse(x, y) + self._eps_square)
        return reduce_data(loss, reduction=self.reduction)

import torch

from dl_toolkit.modules.toolkit_module import ToolkitModule


class IdentityLoss(ToolkitModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.register_buffer("zero_loss", torch.zeros([], dtype=torch.float32))

    def forward(self, *args, **kwargs):
        return self.zero_loss

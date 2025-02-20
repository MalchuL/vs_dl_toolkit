import torch
import torch.nn as nn


class IdentityLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(self)
        self.register_buffer("zero_loss", torch.zeros([], dtype=torch.float32))

    def forward(self, *args, **kwargs):
        return self.zero_loss

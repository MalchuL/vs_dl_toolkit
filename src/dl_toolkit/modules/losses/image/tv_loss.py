import random
import torch
import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self, shift_size=1, reduction='mean'):
        super().__init__()
        assert shift_size >= 1
        self.shift_size = shift_size
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f'Unknown reduction {reduction}')
        self.reduction = reduction

    def forward(self, x):
        shift = random.randint(1, self.shift_size)
        N, C, H, W = x.shape
        h_tv = torch.pow((x[:, :, shift:, :] - x[:, :, :H - shift, :]), 2)
        w_tv = torch.pow((x[:, :, :, shift:] - x[:, :, :, :W - shift]), 2)
        if self.reduction == 'mean':
            return (h_tv.mean() + w_tv.mean())
        elif self.reduction == 'sum':
            return (h_tv.sum() + w_tv.sum()) / 2
        elif self.reduction == 'none':
            return (h_tv.flatten(1) + w_tv.flatten(1))
        else:
            raise ValueError(f'Unknown reduction {self.reduction}')

import pytest
import torch

from dl_toolkit.modules.losses import KLDivergenceLoss


def test_kl():

    loss = KLDivergenceLoss()
    x = torch.randn(20, 8, 256, 256)
    x = torch.clamp(x, -.1, .1)
    # x[:, 2:] = 0
    # print((x < 0).float().mean())
    loss_value = loss(x).item()
    print(loss_value)
    with pytest.raises(AssertionError):
        # Wrong number of channels, must be divisible by 2
        x = torch.randn(10, 3, 256, 256)
        loss_value = loss(x).item()

def test_kl_custom_params():

    loss = KLDivergenceLoss()
    mean, logvar = torch.randn(10, 2, 256, 256),  torch.randn(10, 2, 256, 256)
    loss_value = loss((mean, logvar)).item()
    print(loss_value)
    with pytest.raises(AssertionError):
        # Wrong number of channels must be same
        mean, logvar = torch.randn(10, 2, 256, 256), torch.randn(10, 1, 256, 256)
        loss_value = loss((mean, logvar)).item()

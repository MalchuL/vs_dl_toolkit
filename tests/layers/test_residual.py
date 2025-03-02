import torch
from torch import nn

from dl_toolkit.modules.layers.residual import Residual


def test_residual():
    res = Residual(nn.Conv2d(16, 16, 3, padding=1))
    x = torch.randn(1, 16, 32, 32)
    out = res(x)
    assert torch.allclose(out, x), "Default behavior must be same as input x"

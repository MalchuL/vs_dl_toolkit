import pytest
import torch
from torch import nn

from dl_toolkit.modules.layers.conv.downsample.conv_downsample import (
    AvgPoolNorm,
    BlurPool,
    Downsample,
)


def test_blurpool_shape():
    pool = BlurPool(3)
    x = torch.randn(2, 3, 32, 32)
    assert pool(x).shape == (2, 3, 16, 16)


def test_downsample_channels():
    block = Downsample(3, 64)
    x = torch.randn(2, 3, 32, 32)
    assert block(x).shape == (2, 64, 16, 16)


@pytest.mark.parametrize("pool_type", ["avg", "blur"])
@pytest.mark.parametrize("filt_size", [3, 5])
def test_variance_preservation(pool_type, filt_size):
    if pool_type == "blur":
        block = BlurPool(64, filt_size=filt_size)
    else:
        block = AvgPoolNorm()
    x = torch.randn(256, 64, 32, 32)
    out = block(x)
    var_in = x.var()
    var_out = out.var()
    print(var_in, var_out)
    assert out.shape[2] == x.shape[2] // 2
    assert torch.allclose(var_in, var_out, rtol=0.1)


def test_gradient_flow():
    block = Downsample(3)
    x = torch.randn(2, 3, 32, 32).requires_grad_()
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_5x5_blur():
    pool = BlurPool(3, filt_size=5)
    x = torch.randn(2, 3, 32, 32)
    assert pool(x).shape == (2, 3, 16, 16)

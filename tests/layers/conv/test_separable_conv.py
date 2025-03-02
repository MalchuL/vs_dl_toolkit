import pytest
import torch
from torch import nn

from dl_toolkit.modules.layers.conv.separable_conv2d import SeparableConv2d
from dl_toolkit.modules.utils.init_utils import init_net

# Assume BIAS and PADDING_MODE are defined in constants
BIAS = True
PADDING_MODE = "zeros"


@pytest.fixture
def sample_input():
    return torch.randn(2, 3, 32, 32)  # Batch of 2 3-channel 32x32 images


def test_default_mode_construction():
    conv = SeparableConv2d(3, 16, sep_mode="default")

    # Should have depthwise + pointwise layers
    assert len(conv.module) == 2
    assert isinstance(conv.module[0], nn.Conv2d)
    assert isinstance(conv.module[1], nn.Conv2d)

    # Verify depthwise parameters
    assert conv.module[0].groups == 3
    assert conv.module[0].out_channels == 3
    assert conv.module[0].bias is None

    # Verify pointwise parameters
    assert conv.module[1].kernel_size == (1, 1)
    assert conv.module[1].out_channels == 16


def test_inception_mode_construction():
    conv = SeparableConv2d(3, 16, sep_mode="inception")

    assert len(conv.module) == 2
    assert isinstance(conv.module[0], nn.Conv2d)
    assert isinstance(conv.module[1], nn.Conv2d)

    # First layer should be pointwise
    assert conv.module[0].kernel_size == (1, 1)
    assert conv.module[0].out_channels == 16

    # Second layer should be depthwise
    assert conv.module[1].groups == 16
    assert conv.module[1].out_channels == 16


def test_kernel_size_1_construction():
    conv = SeparableConv2d(3, 16, kernel_size=1)
    assert len(conv.module) == 1
    assert isinstance(conv.module[0], nn.Conv2d)
    assert conv.module[0].kernel_size == (1, 1)


def test_output_shape(sample_input):
    conv = SeparableConv2d(3, 16, kernel_size=2, stride=2)
    output = conv(sample_input)
    assert output.shape == (2, 16, 16, 16)  # (32-3+2*1)/2 +1 = 16


def test_padding_default():
    conv = SeparableConv2d(3, 16, kernel_size=5, padding=None)
    assert conv.padding == 2  # 5//2


def test_variance():
    x = torch.randn(2, 32, 32, 32)
    conv = SeparableConv2d(32, 128, kernel_size=5, padding=None)
    init_net(conv, "kaiming_uniform")
    out = conv(x)
    assert torch.allclose(out.std(), x.std(), atol=0.1)


def test_bias_handling():
    # Test with bias disabled
    conv = SeparableConv2d(3, 16, bias=False)
    assert conv.module[0].bias is None
    assert conv.module[1].bias is None

    # Test with bias enabled (default)
    conv = SeparableConv2d(3, 16, bias=True)
    assert conv.module[0].bias is None  # Depthwise never has bias
    assert conv.module[1].bias is not None


def test_invalid_sep_mode():
    with pytest.raises(ValueError):
        SeparableConv2d(3, 16, sep_mode="invalid")


def test_gradient_propagation(sample_input):
    conv = SeparableConv2d(3, 16)
    output = conv(sample_input)
    loss = output.sum()
    loss.backward()
    # Verify parameters have gradients
    for param in conv.parameters():
        assert param.grad is not None

import pytest
import torch
from torch import nn

from dl_toolkit.modules.layers.conv.resnet.res_block_v2 import ResidualBlockV2


def test_dilation_gt1_raises_error():
    with pytest.raises(NotImplementedError):
        ResidualBlockV2(64, 64, dilation=2)

@pytest.mark.parametrize("in_c, out_c, stride, has_downsample", [
    (64, 64, 1, False),
    (64, 128, 1, True),
    (64, 64, 2, True),
    (128, 128, 2, True),
])
def test_downsample_creation(in_c, out_c, stride, has_downsample):
    block = ResidualBlockV2(in_c, out_c, stride=stride)
    if has_downsample:
        assert isinstance(block.downsample, nn.Conv2d)
    else:
        assert isinstance(block.downsample, nn.Identity)

@pytest.mark.parametrize("norm_layer, zero_init", [
    ('batch', True),
    ('none', True),
    ('batch', False),
])
def test_zero_init_residual(norm_layer, zero_init):
    block = ResidualBlockV2(64, 64, norm_layer=norm_layer, zero_init_residual=zero_init)
    if zero_init:
        if norm_layer == 'batch':
            assert torch.all(block.conv2.norm.weight == 0)
            assert torch.all(block.conv2.norm.bias == 0)
        else:
            conv = block.conv2.conv
            if isinstance(conv, nn.Sequential):
                conv = conv[-1]
            assert torch.all(conv.weight == 0)
            assert torch.all(conv.bias == 0)

        x = torch.randn(1, 64, 32, 32)
        out = block(x)
        assert torch.allclose(out, x)
    else:
        if norm_layer == 'batch':
            assert not torch.all(block.conv2.norm.weight == 0)
        else:
            conv = block.conv2.conv
            if isinstance(conv, nn.Sequential):
                conv = conv[-1]
            assert not torch.all(conv.weight == 0)

@pytest.mark.parametrize("separable", [True, False])
@pytest.mark.parametrize("norm_layer", ['batch', 'none'])
def test_zero_init_separable(separable, norm_layer):
    block = ResidualBlockV2(64, 64, norm_layer=norm_layer, separable=separable, zero_init_residual=True)
    x = torch.randn(1, 64, 32, 32)
    out = block(x)
    assert torch.allclose(out, x)


@pytest.mark.parametrize("in_c, out_c, stride, input_shape, expected_shape", [
    (64, 64, 1, (1, 64, 32, 32), (1, 64, 32, 32)),
    (64, 128, 2, (1, 64, 32, 32), (1, 128, 16, 16)),
    (128, 256, 2, (2, 128, 64, 64), (2, 256, 32, 32)),
])
def test_forward_shape(in_c, out_c, stride, input_shape, expected_shape):
    block = ResidualBlockV2(in_c, out_c, stride=stride)
    x = torch.randn(*input_shape)
    out = block(x)
    assert out.shape == torch.Size(expected_shape)

def test_forward_output_is_sum():
    block = ResidualBlockV2(64, 64)
    x = torch.randn(1, 64, 32, 32)
    out = block(x)
    residual = block.downsample(x)
    expected = residual + block.conv2(block.conv1(x))
    assert torch.allclose(out, expected)
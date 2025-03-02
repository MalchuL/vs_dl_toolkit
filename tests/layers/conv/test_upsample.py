import pytest
import torch

from dl_toolkit.modules.layers.conv.upsample.conv_upsample import Upsample


def test_inheritance():
    """Test if Upsample inherits from ToolkitModule."""
    assert issubclass(Upsample, torch.nn.Module)  # Assuming ToolkitModule is a nn.Module subclass


@pytest.mark.parametrize("in_channels, out_channels, depthwise, input_size", [
    (3, 3, False, (4, 4)),  # Basic case
    (3, 6, False, (5, 5)),  # Channel expansion
    (4, 8, True, (3, 3)),  # Valid depthwise (8 % 4 == 0)
    (2, 2, True, (6, 6)),  # Depthwise same channels
    (1, 1, True, (2, 2)),  # Minimum channels
])
def test_output_shape(in_channels, out_channels, depthwise, input_size):
    """Test if output shape matches expectations after upsampling."""
    if depthwise and (out_channels % in_channels != 0):
        pytest.skip("Invalid depthwise configuration for this test case")

    model = Upsample(
        in_channels=in_channels,
        out_channels=out_channels,
        depthwise=depthwise,
    )

    x = torch.randn(2, in_channels, *input_size)  # Batch size 2
    output = model(x)

    expected_shape = (2, out_channels, input_size[0] * 2, input_size[1] * 2)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"


@pytest.mark.parametrize("depthwise, groups", [
    (True, 4),  # Depthwise should set groups=in_channels
    (False, 1),  # Regular convolution uses groups=1
])
def test_conv_groups(depthwise, groups):
    """Test depthwise convolution group configuration."""
    in_channels = 4
    model = Upsample(
        in_channels=in_channels,
        depthwise=depthwise,
    )
    assert model.conv.groups == groups, "Incorrect convolution groups"


@pytest.mark.parametrize("interpolate_mode", ["nearest", "bilinear"])
def test_interpolation_modes(interpolate_mode):
    """Test different interpolation modes."""
    model = Upsample(
        in_channels=3,
        interpolate_mode=interpolate_mode,
    )
    x = torch.randn(1, 3, 4, 4)
    output = model(x)
    assert output.shape == (1, 3, 8, 8), f"Failed for mode {interpolate_mode}"


def test_default_out_channels():
    """Test default out_channels assignment."""
    in_channels = 5
    model = Upsample(in_channels=in_channels)
    assert model.conv.out_channels == in_channels, "Default out_channels should match in_channels"


def test_convolution_bias():
    """Test convolution layer has no bias."""
    model = Upsample(in_channels=3)
    assert model.conv.bias is None, "Conv layer should have no bias"


def test_depthwise_error():
    """Test invalid depthwise configuration raises error."""
    with pytest.raises(ValueError, match="divisible"):
        Upsample(
            in_channels=3,
            out_channels=4,  # 4 not divisible by 3
            depthwise=True,
        )


def test_odd_size_upsampling():
    """Test upsampling with odd spatial dimensions."""
    model = Upsample(in_channels=3)
    x = torch.randn(1, 3, 3, 3)
    output = model(x)
    assert output.shape == (1, 3, 6, 6), "Odd input size handling failed"

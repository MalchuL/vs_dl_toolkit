import pytest
import torch
import numpy as np
from torch import nn

from dl_toolkit.modules.layers.conv.sobel import SobelFilter


@pytest.fixture
def sample_image():
    return torch.randn(2, 3, 32, 32)  # Batch of 2 RGB images


def test_initialization():
    sf = SobelFilter(k_sobel=3)
    assert sf.sobel_filter_x.shape == (1, 1, 3, 3)
    assert sf.sobel_filter_y.shape == (1, 1, 3, 3)
    assert sf.reduction_weight.shape == (1, 3, 1, 1)
    assert isinstance(sf.padding, nn.ReflectionPad2d)


def test_output_channels(sample_image):
    # Test only_edges=True
    sf = SobelFilter(only_edges=True)
    output = sf(sample_image)
    assert output.shape[1] == 1  # Single channel for edge magnitude

    # Test only_edges=False
    sf = SobelFilter(only_edges=False)
    output = sf(sample_image)
    assert output.shape[1] == 2  # Two channels for gradients


def test_padding_behavior():
    # With padding
    sf = SobelFilter(use_padding=True, k_sobel=3)
    x = torch.randn(1, 3, 5, 5)
    assert sf(x).shape[2:] == (5, 5)

    # Without padding
    sf = SobelFilter(use_padding=False, k_sobel=3)
    assert sf(x).shape[2:] == (3, 3)  # 5-2=3


def test_edge_magnitude_calculation():
    sf = SobelFilter(only_edges=False)
    # Create vertical edge image
    x = torch.zeros(1, 3, 5, 5)
    x[:, :, :, 1] = 1  # Vertical line
    output = sf(x)
    # Expect strong edges in middle column
    assert torch.all(output[0, 0, :, 2] > 0.5)


def test_uniform_image():
    sf = SobelFilter(only_edges=True)
    uniform = torch.ones(1, 3, 5, 5)
    output = sf(uniform)
    # Gradients should be near zero in uniform areas
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-4)


def test_grayscale_conversion():
    sf = SobelFilter()
    rgb = torch.tensor([[[0.5]], [[0.3]], [[0.2]]]).unsqueeze(0)  # (1,3,1,1)
    gray = sf.rgb2gray(rgb)
    expected = 0.299 * 0.5 + 0.587 * 0.3 + 0.114 * 0.2
    assert torch.allclose(gray, torch.tensor([[[[expected]]]]))


def test_kernel_values():
    kernel = SobelFilter.get_sobel_kernel(3)
    assert np.allclose(kernel, [
        [-0.5, 0, 0.5],
        [-1.0, 0, 1.0],
        [-0.5, 0, 0.5]
    ], atol=0.1)
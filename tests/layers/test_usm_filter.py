"""Image sharpening module with Unsharp Mask (USM) implementation."""

import pytest
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from dl_toolkit.modules.layers.conv.representation.usm_sharp import USMSharp


def test_usm_output_shape():
    """Test output shape matches input dimensions."""
    module = USMSharp(radius=5)
    img = torch.rand(2, 3, 64, 64)
    output = module(img)
    assert output.shape == img.shape

def test_kernel_properties():
    """Verify Gaussian kernel properties."""
    radius = 5
    module = USMSharp(radius=radius)
    assert module.kernel.shape == (1, radius, radius)
    assert torch.allclose(module.kernel.sum(), torch.tensor(1.0), atol=1e-5)

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dtype_handling(dtype):
    """Test different input data types."""
    module = USMSharp()
    img = torch.rand(1, 3, 32, 32).to(dtype)
    module = module.to(dtype)
    output = module(img)
    assert output.dtype == dtype

def test_device_consistency():
    """Test module preserves input device."""
    module = USMSharp()
    for device in ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']:
        img = torch.rand(1, 3, 64, 64).to(device)
        module.to(device)
        output = module(img)
        assert output.device == img.device

def test_value_clamping():
    """Verify output values stay within [0,1] range."""
    module = USMSharp()
    img = torch.rand(1, 3, 32, 32) * 1.5  # Create values >1
    output = module(img)
    assert torch.all(output >= 0) and torch.all(output <= 1)

def test_threshold_behavior():
    """Test mask thresholding effect."""
    module = USMSharp(radius=3)
    img = torch.ones(1, 3, 16, 16) * (10/255)  # Below threshold
    output = module(img, threshold=10)
    assert torch.allclose(output, img)

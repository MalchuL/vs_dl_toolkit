"""Guided filter implementation tests and documentation."""

import pytest
import torch
from torch import nn

from dl_toolkit.modules.layers.conv.representation.guided_filter import GuidedFilter


class TestGuidedFilter:
    """Test suite for GuidedFilter module."""

    @pytest.mark.parametrize(
        "input_shape,r",
        [
            ((2, 3, 64, 64), 1),
            ((1, 3, 32, 32), 3),
            ((4, 3, 128, 128), 5),
        ],
    )
    def test_output_shape(self, input_shape, r):
        """Verify output maintains input dimensions.

        Args:
            input_shape (tuple): Input tensor shape (NCHW)
            r (int): Filter radius parameter
        """
        module = GuidedFilter(r=r, input_channels=3)
        x = torch.rand(*input_shape)
        output = module(x, x)
        assert output.shape == x.shape

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_device_consistency(self, device):
        """Ensure module preserves input device.

        Args:
            device (str): Target device for test
        """
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        module = GuidedFilter().to(device)
        x = torch.rand(2, 3, 32, 32).to(device)
        output = module(x, x)
        assert output.device == x.device

    def test_gradient_flow(self):
        """Verify backward pass computes gradients."""
        module = GuidedFilter()
        x = torch.rand(2, 3, 64, 64, requires_grad=True)
        y = torch.rand_like(x)
        output = module(x, y)
        loss = output.mean()
        loss.backward()
        assert x.grad is not None

    def test_numerical_stability(self):
        """Check for valid numerical outputs."""
        module = GuidedFilter(eps=1e-6)
        x = torch.zeros(1, 3, 16, 16)
        output = module(x, x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize("input_channels", [1, 3, 5])
    def test_channel_handling(self, input_channels):
        """Test different input channel configurations.

        Args:
            input_channels (int): Number of input channels
        """
        module = GuidedFilter(input_channels=input_channels)
        x = torch.rand(2, input_channels, 32, 32)
        output = module(x, x)
        assert output.shape == x.shape

    def test_dtype_consistency(self):
        """Verify output dtype matches input."""
        module = GuidedFilter()
        for dtype in [torch.float16, torch.float32, torch.float64]:
            module.to(dtype)
            x = torch.rand(2, 3, 16, 16).to(dtype)
            output = module(x, x)
            assert output.dtype == dtype

    def test_invalid_input_shapes(self):
        """Test shape mismatch detection."""
        module = GuidedFilter()
        x = torch.rand(2, 3, 32, 32)
        y = torch.rand(2, 3, 64, 64)
        with pytest.raises(RuntimeError):
            module(x, y)

    def test_box_filter_application(self):
        """Validate box filter kernel selection."""
        module = GuidedFilter(r=2, input_channels=3)
        x = torch.rand(1, 3, 16, 16)

        # Test multi-channel filter
        multi_out = module.box_filter(x)
        assert multi_out.shape == x.shape

        # Test single-channel filter
        single_out = module.box_filter(x[:, :1], channels=1)
        assert single_out.shape == (1, 1, 16, 16)

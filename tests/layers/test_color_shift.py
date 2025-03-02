"""Color shift module tests and documentation."""

import pytest
import torch

from dl_toolkit.modules.layers.conv.representation.color_shift import ColorShift


class TestColorShift:
    """Test suite for ColorShift module."""

    @pytest.mark.parametrize(
        "is_repeat,expected_channels",
        [
            (True, 3),
            (False, 1),
        ],
    )
    def test_output_shape(self, is_repeat, expected_channels):
        """Test output shape matches expectations.

        Args:
            is_repeat (bool): Whether to repeat output channels
            expected_channels (int): Expected number of output channels
        """
        module = ColorShift(is_repeat=is_repeat)
        input_tensor = torch.randn(2, 3, 32, 32)
        output = module(input_tensor)
        assert output.shape == (2, expected_channels, 32, 32)

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_device_consistency(self, device):
        """Test module preserves input device.

        Args:
            device (str): Target device for test
        """
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        module = ColorShift().to(device)
        input_tensor = torch.randn(2, 3, 16, 16).to(device)
        output = module(input_tensor)
        assert output.device == input_tensor.device

    def test_gradient_flow(self):
        """Test backward pass through module."""
        module = ColorShift()
        input_tensor = torch.randn(2, 3, 32, 32, requires_grad=True)
        output = module(input_tensor)
        loss = output.mean()
        loss.backward()
        assert input_tensor.grad is not None

    def test_invalid_weight_mode(self):
        """Test invalid weight mode raises error."""
        with pytest.raises(ValueError):
            ColorShift(weight_mode="invalid")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_consistency(self, dtype):
        """Test output dtype matches input.

        Args:
            dtype (torch.dtype): Input data type
        """
        module = ColorShift()
        input_tensor = torch.randn(2, 3, 16, 16).to(dtype=dtype)
        output = module(input_tensor)
        assert output.dtype == dtype

    def test_get_num_channels(self):
        """Test channel count reporting."""
        module = ColorShift(is_repeat=True)
        assert module.get_num_channels() == 3
        module.is_repeat = False
        assert module.get_num_channels() == 1

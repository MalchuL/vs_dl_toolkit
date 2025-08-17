import pytest
import torch
import torch.nn as nn

from dl_toolkit.modules.layers.conv.representation.style_representation import StyleRepresentation


@pytest.fixture
def input_tensor():
    """Create a sample input tensor."""
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def invalid_input():
    """Create an invalid input tensor (wrong number of channels)."""
    return torch.randn(2, 4, 32, 32)


class TestStyleRepresentation:
    def test_empty_layers(self, input_tensor):
        """Test StyleRepresentation with no layers."""
        with pytest.raises(ValueError, match="At least one layer must be specified"):
            StyleRepresentation()

    def test_single_layer(self, input_tensor):
        """Test each individual layer type."""
        layer_tests = [
            ('identity', 3),  # Original 3 channels
            ('color_shift', 3),  # 3 channels output
            ('sobel', 3),  # 1 channel repeated to 3
            ('surface', 3),  # 3 channels output
        ]

        for layer_name, expected_channels in layer_tests:
            model = StyleRepresentation(layers=(layer_name,))
            out = model(input_tensor)
            assert out.shape == (2, expected_channels, 32, 32), f"Failed for layer {layer_name}"
            assert model.channels == expected_channels

    def test_multiple_layers(self, input_tensor):
        """Test combination of multiple layers."""
        layers = ('identity', 'color_shift', 'sobel')
        model = StyleRepresentation(layers=layers)
        out = model(input_tensor)
        expected_channels = len(layers) * 3  # Each layer outputs 3 channels
        assert out.shape == (2, expected_channels, 32, 32)
        assert model.channels == expected_channels

    def test_with_coordinates(self, input_tensor):
        """Test with coordinate channels added."""
        layers = ('identity', 'color_shift')
        model = StyleRepresentation(layers=layers, add_coords=True)
        out = model(input_tensor)
        channels_per_layer = 5  # 3 original + 2 coordinate channels
        expected_channels = len(layers) * channels_per_layer
        assert out.shape == (2, expected_channels, 32, 32)
        assert model.channels == expected_channels

    def test_surface_layer(self, input_tensor):
        """Test surface layer with different parameters."""
        model = StyleRepresentation(layers=('surface',), r=3, eps=1e-2)
        out = model(input_tensor)
        assert out.shape == (2, 3, 32, 32)
        
        # Test that different parameters produce different outputs
        model2 = StyleRepresentation(layers=('surface',), r=7, eps=1e-1)
        out2 = model2(input_tensor)
        assert not torch.allclose(out, out2)

    def test_invalid_input_channels(self, invalid_input):
        """Test handling of invalid input channel count."""
        model = StyleRepresentation(layers=('identity',))
        with pytest.raises(ValueError, match="Input tensor must have 3 channels"):
            model(invalid_input)

    def test_all_layers_combination(self, input_tensor):
        """Test all possible layers together."""
        all_layers = ('identity', 'color_shift', 'sobel', 'surface')
        model = StyleRepresentation(layers=all_layers, add_coords=True)
        out = model(input_tensor)
        channels_per_layer = 5  # 3 original + 2 coordinate channels
        expected_channels = len(all_layers) * channels_per_layer
        assert out.shape == (2, expected_channels, 32, 32)
        assert model.channels == expected_channels

    def test_layer_output_consistency(self, input_tensor):
        """Test that layer outputs are consistent and properly concatenated."""
        model = StyleRepresentation(layers=('identity', 'sobel'))
        out = model(input_tensor)
        
        # First 3 channels should be identical to input (identity layer)
        assert torch.allclose(out[:, :3], input_tensor)
        
        # Next 3 channels should be identical (sobel output repeated)
        assert torch.allclose(out[:, 3:4], out[:, 4:5])
        assert torch.allclose(out[:, 4:5], out[:, 5:6])

    @pytest.mark.parametrize("weight_mode", ["uniform", "normal"])
    def test_color_shift_modes(self, input_tensor, weight_mode):
        """Test different weight modes for color shift layer."""
        model = StyleRepresentation(layers=('color_shift',), weight_mode=weight_mode)
        out = model(input_tensor)
        assert out.shape == (2, 3, 32, 32)

    def test_output_range(self, input_tensor):
        """Test that outputs are in a reasonable range."""
        # Normalize input to [0, 1] for this test
        input_tensor = torch.sigmoid(input_tensor)
        
        model = StyleRepresentation(layers=('identity', 'color_shift', 'sobel', 'surface'))
        out = model(input_tensor)
        
        # Check if outputs are in a reasonable range
        assert out.min() >= -2 and out.max() <= 2, "Output values are in unexpected range"



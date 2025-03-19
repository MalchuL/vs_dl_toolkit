import pytest
import torch.nn as nn
from functools import partial

from dl_toolkit.modules.layers.norm_layers.group_norm import GroupNorm, GroupNorm8
from dl_toolkit.modules.layers.norm_layers import get_norm_layer


@pytest.mark.parametrize("norm_type, expected_cls, expected_params", [
    ("batch", nn.BatchNorm2d, {"affine": True}),
    ("instance", nn.InstanceNorm2d, {"affine": False, "track_running_stats": False}),
    ("group", GroupNorm, {}),
    ("group8", GroupNorm8, {}),
    ("none", nn.Identity, {}),
    (None, nn.Identity, {}),
    (nn.LayerNorm, nn.LayerNorm, {}),
])
def test_valid_norm_types(norm_type, expected_cls, expected_params):
    """Test valid norm_type inputs return the correct class/partial with parameters."""
    result = get_norm_layer(norm_type)

    if isinstance(result, partial):
        # Check partial configuration
        assert result.func == expected_cls
        for key, value in expected_params.items():
            assert result.keywords.get(key) == value
    else:
        # Direct class return
        assert result == expected_cls


def test_custom_module_subclass():
    """Test passing a custom nn.Module subclass returns it directly."""

    class CustomNorm(nn.Module):
        pass

    assert get_norm_layer(CustomNorm) == CustomNorm


def test_invalid_norm_type_raises_error():
    """Test invalid norm_type strings raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        get_norm_layer("invalid_norm")


def test_layer_instantiation():
    """Test the returned layer can be instantiated with correct parameters."""
    # Test BatchNorm2d configuration
    batch_norm = get_norm_layer("batch")
    layer = batch_norm(64)
    assert isinstance(layer, nn.BatchNorm2d)
    assert layer.affine  # Parameter should be True

    # Test InstanceNorm2d configuration
    instance_norm = get_norm_layer("instance")
    layer = instance_norm(64)
    assert isinstance(layer, nn.InstanceNorm2d)
    assert not layer.affine
    assert not layer.track_running_stats

    # Test GroupNorm (requires num_groups)
    group_norm_cls = get_norm_layer("group")
    layer = group_norm_cls(num_channels=64)  # Example usage
    assert isinstance(layer, GroupNorm)

    lambda_norm = lambda x: nn.InstanceNorm2d(x)
    lambda_norm_cls = get_norm_layer(lambda_norm)
    layer = lambda_norm_cls(64)
    assert isinstance(layer, nn.InstanceNorm2d)

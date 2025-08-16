"""Tests for VGG feature extractor module."""

import pytest
import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights

from dl_toolkit.modules.feature_extractors.vgg_features import (
    NETWORKS_CONFIGS,
    PaddingType,
    VGGFeatures,
)


@pytest.fixture
def input_tensor():
    """Create a sample input tensor normalized to [0, 1] range."""
    torch.manual_seed(42)
    return torch.rand(2, 3, 64, 64)


@pytest.fixture
def vgg_features():
    """Create a default VGG feature extractor."""
    return VGGFeatures(layers=[2, 5, 7], network="vgg16")


def test_vgg_features_initialization():
    """Test VGGFeatures initialization with different parameters."""
    # Test default initialization
    extractor = VGGFeatures(layers=[2, 5])
    assert extractor.layers == [2, 5]
    assert isinstance(extractor.clipper, nn.Identity)

    # Test with z-clipping
    extractor = VGGFeatures(layers=[2], z_clipping=2.0)
    assert extractor.z_clipping == 2.0
    assert not isinstance(extractor.clipper, nn.Identity)

    # Test with reflection padding
    extractor = VGGFeatures(layers=[2], padding_type=PaddingType.REFLECT)
    modules = list(extractor.perception.modules())
    conv_layers = [m for m in modules if isinstance(m, nn.Conv2d)]
    assert all(layer.padding_mode == "reflect" for layer in conv_layers)

    # Test with valid padding
    extractor = VGGFeatures(layers=[2], padding_type=PaddingType.VALID)
    modules = list(extractor.perception.modules())
    conv_layers = [m for m in modules if isinstance(m, nn.Conv2d)]
    assert all(
        (isinstance(layer.padding, (tuple, int)) and layer.padding == 0)
        or (isinstance(layer.padding, tuple) and all(p == 0 for p in layer.padding))
        for layer in conv_layers
    )


def test_vgg_features_invalid_initialization():
    """Test VGGFeatures initialization with invalid parameters."""
    # Test empty layers list
    with pytest.raises(AssertionError):
        VGGFeatures(layers=[])

    # Test negative layer index
    with pytest.raises(AssertionError):
        VGGFeatures(layers=[-1])

    # Test invalid network name
    with pytest.raises(ValueError):
        VGGFeatures(layers=[0], network="invalid_network")

    # Test layer index exceeding network depth
    max_layer = NETWORKS_CONFIGS["vgg16"].max_layer
    with pytest.raises(AssertionError):
        VGGFeatures(layers=[max_layer + 1], network="vgg16")


def test_vgg_features_forward(input_tensor, vgg_features):
    """Test forward pass of VGGFeatures."""
    # Test basic forward pass
    features = vgg_features(input_tensor)
    assert isinstance(features, dict)
    assert set(features.keys()) == {2, 5, 7}
    assert all(isinstance(feat, torch.Tensor) for feat in features.values())
    assert all(feat.requires_grad is False for feat in features.values())

    # Check output shapes (should decrease due to pooling)
    shapes = [feat.shape[-1] for feat in features.values()]
    assert shapes[0] >= shapes[1] >= shapes[2]  # Spatial dimensions should decrease


def test_vgg_features_normalization(input_tensor, vgg_features):
    """Test input normalization in VGGFeatures."""
    # Get normalized input directly
    normalized = (input_tensor - vgg_features.mean) / vgg_features.std

    # Run forward pass with hooks to capture first layer input
    first_layer_input = None

    def hook(module, input_feat, output):
        nonlocal first_layer_input
        first_layer_input = input_feat[0].clone()

    first_conv = next(
        m for m in vgg_features.perception.modules() if isinstance(m, nn.Conv2d)
    )
    handle = first_conv.register_forward_hook(hook)
    
    vgg_features(input_tensor)
    handle.remove()

    # Compare normalized inputs
    assert torch.allclose(normalized, first_layer_input)


def test_vgg_features_z_clipping():
    """Test feature clipping functionality."""
    # Create feature extractor with aggressive clipping
    extractor = VGGFeatures(layers=[2], z_clipping=1.0)
    
    # Create input with extreme values
    input_tensor = torch.randn(2, 3, 64, 64) * 10  # Large variance
    input_tensor = torch.sigmoid(input_tensor)  # Normalize to [0, 1]
    
    features = extractor(input_tensor)
    
    # Check if features are clipped
    for feat in features.values():
        # For z_score=1.0, ~68% of values should be within ±1 std
        within_bounds = (feat.abs() <= 1.0).float().mean()
        assert within_bounds > 0.0


def test_vgg_features_different_networks():
    """Test VGGFeatures with different network architectures."""
    input_tensor = torch.rand(2, 3, 64, 64)
    
    for network in ["vgg16", "vgg19", "vgg19_bn"]:
        extractor = VGGFeatures(layers=[2], network=network)
        features = extractor(input_tensor)
        
        # Basic checks
        assert isinstance(features, dict)
        assert 2 in features
        assert isinstance(features[2], torch.Tensor)
        
        # Check if batch norm layers are present when expected
        has_bn = any(
            isinstance(m, nn.BatchNorm2d)
            for m in extractor.perception.modules()
        )
        assert has_bn == (network == "vgg19_bn")


def test_vgg_features_padding_modes(input_tensor):
    """Test different padding modes affect output size."""
    layer_idx = 2
    sizes = {}
    
    for padding in PaddingType:
        extractor = VGGFeatures(
            layers=[layer_idx],
            padding_type=padding
        )
        features = extractor(input_tensor)
        sizes[padding] = features[layer_idx].shape[-2:]
    
    # VALID padding should result in smaller spatial dimensions
    assert sizes[PaddingType.VALID] < sizes[PaddingType.ZEROS]
    # ZEROS and REFLECT should maintain the same size
    assert sizes[PaddingType.ZEROS] == sizes[PaddingType.REFLECT]


def test_vgg_features_pretrained_weights():
    """Test if pretrained weights are loaded correctly."""
    extractor = VGGFeatures(layers=[2], network="vgg16")
    
    # Get the first conv layer
    first_conv = next(
        m for m in extractor.perception.modules() if isinstance(m, nn.Conv2d)
    )
    
    # Load VGG16 directly
    vgg = torch.hub.load(
        'pytorch/vision:v0.10.0',
        'vgg16',
        weights=VGG16_Weights.IMAGENET1K_V1
    )
    ref_first_conv = next(
        m for m in vgg.modules() if isinstance(m, nn.Conv2d)
    )
    
    # Compare weights of first conv layer
    assert torch.allclose(first_conv.weight, ref_first_conv.weight)
    assert torch.allclose(first_conv.bias, ref_first_conv.bias)
"""VGG Feature Extractor Module.

This module provides functionality to extract features from different variants of VGG networks
(VGG16, VGG19, VGG19 with batch normalization). It supports various padding modes and feature
normalization options.

Key Features:
    - Support for VGG16, VGG19, and VGG19-BN architectures
    - Multiple padding modes (zeros, reflect, valid)
    - Feature normalization and clipping
    - Selective layer feature extraction
    - ImageNet pretrained weights

Example:
    >>> extractor = VGGFeatures(
    ...     layers=[2, 7, 14],  # Extract features from these layers
    ...     network="vgg16",    # Use VGG16 architecture
    ...     padding_type=PaddingType.REFLECT,  # Use reflection padding
    ...     z_clipping=2.0      # Clip features at 2 standard deviations
    ... )
    >>> image = torch.randn(1, 3, 224, 224)  # Input image (normalized to [0,1])
    >>> features = extractor(image)  # Dictionary of layer features
"""

from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import List, Mapping, Callable, Dict

import torch
import torch.nn as nn
from torchvision.models import (
    VGG,
    VGG16_Weights,
    VGG19_BN_Weights,
    VGG19_Weights,
    WeightsEnum,
    vgg16,
    vgg19,
    vgg19_bn,
)

from dl_toolkit.modules.layers.clipper import ClipperChannelwise2D
from dl_toolkit.modules.toolkit_module import ToolkitModule

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB channel means
IMAGENET_STD = [0.229, 0.224, 0.225]   # RGB channel standard deviations


@dataclass
class ModelConfig:
    """Configuration class for VGG model variants.
    
    Attributes:
        max_layer (int): Maximum layer index available in the model
        weights (WeightsEnum): Pretrained weights configuration
        func (Callable): Constructor function for the model
        mean (List[float]): Channel-wise mean values for input normalization
        std (List[float]): Channel-wise standard deviation values for input normalization
    """
    max_layer: int
    weights: WeightsEnum
    func: Callable
    mean: List[float] = field(default_factory=lambda: IMAGENET_MEAN)
    std: List[float] = field(default_factory=lambda: IMAGENET_STD)


NETWORKS_CONFIGS: Mapping[str, ModelConfig] = {
    "vgg16": ModelConfig(max_layer=30, weights=VGG16_Weights.IMAGENET1K_V1, func=vgg16),
    "vgg19": ModelConfig(max_layer=36, weights=VGG19_Weights.IMAGENET1K_V1, func=vgg19),
    "vgg19_bn": ModelConfig(max_layer=52, weights=VGG19_BN_Weights.IMAGENET1K_V1, func=vgg19_bn),
}


class PaddingType(Enum):
    """Padding modes for VGG feature extraction.

    Available modes:
        ZEROS: Standard zero padding (default VGG behavior)
        REFLECT: Reflection padding at boundaries (reduces border artifacts)
        VALID: No padding (output size will be reduced)
    """
    ZEROS = 0   # Default VGG padding - pads with zeros
    REFLECT = 1 # Replaces zero padding with reflection padding - better for style transfer
    VALID = 2   # NO padding - image will be cropped but no artificial boundaries introduced


class VGGFeatures(ToolkitModule):
    VERSION = "1.0.0"

    def __init__(
        self,
        layers: List[int],
        network: str = "vgg16",
        padding_type: PaddingType = PaddingType.ZEROS,
        z_clipping: float | None = None,
    ):
        """Initialize VGG feature extractor.

        Args:
            layers (List[int]): Layer indices to extract features from. Must be valid indices
                for the chosen network (see NETWORKS_CONFIGS for max values).
            network (str, optional): VGG variant to use. Options: "vgg16", "vgg19", "vgg19_bn".
                Defaults to "vgg16".
            padding_type (PaddingType, optional): Type of padding to use in convolutions.
                Defaults to PaddingType.ZEROS.
            z_clipping (float | None, optional): If provided, clips feature values to within
                this many standard deviations from the mean. Recommended range: [1.5, 3.0].
                Useful for stabilizing feature statistics. Defaults to None.

        Raises:
            ValueError: If network name is invalid or layer indices exceed network depth.
            AssertionError: If layers list is empty or contains negative indices.

        Note:
            Input images should be normalized to [0, 1] range before passing to the model.
            The module will handle ImageNet normalization internally.
        """
        super().__init__()

        assert len(layers) > 0
        assert min(layers) >= 0
        self.layers = list(sorted(set(layers)))

        # Instantiate models
        self.perception = self._get_perception(network, layers)

        # Make normalization of features
        if network not in NETWORKS_CONFIGS:
            raise ValueError(
                f"Unknown network {network}. "
                f"Allowed networks: {list(NETWORKS_CONFIGS.keys())}"
            )
        model_config = NETWORKS_CONFIGS[network]
        self.register_buffer("mean", torch.tensor(model_config.mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(model_config.std).view(1, -1, 1, 1))

        self.padding_type = padding_type
        self._fix_padding(self.perception, self.padding_type)

        self.z_clipping = z_clipping
        if self.z_clipping is None:
            self.clipper = nn.Identity()
        else:
            self.clipper = ClipperChannelwise2D(z_score=self.z_clipping)

    @classmethod
    def _get_network(cls, model_config: ModelConfig) -> VGG:
        """Create a VGG network instance with specified configuration.

        Args:
            model_config (ModelConfig): Configuration specifying network variant and weights.

        Returns:
            VGG: Instantiated VGG model with pretrained weights.
        """
        constructor = model_config.func
        return constructor(weights=model_config.weights)

    def _get_perception(self, network_name: str, layers: List[int]) -> nn.Sequential:
        """Create feature extraction pipeline up to the maximum requested layer.

        Args:
            network_name (str): Name of VGG variant to use.
            layers (List[int]): Layer indices to extract features from.

        Returns:
            nn.Sequential: Sequential module containing VGG layers up to max requested index.

        Raises:
            AssertionError: If requested layer index exceeds network depth.
            ValueError: If network_name is not recognized.
        """
        if network_name not in NETWORKS_CONFIGS:
            raise ValueError(
                f"Unknown network {network_name}. "
                f"Allowed networks: {list(NETWORKS_CONFIGS.keys())}"
            )
        model_config = NETWORKS_CONFIGS[network_name]
        max_layer = max(layers)
        assert max_layer <= model_config.max_layer, (
            "Max layer exceeded for "
            f"network {network_name} "
            f"max {model_config.max_layer}"
        )

        model = self._get_network(model_config)
        perception = list(model.features)[: max_layer + 1]
        perception = nn.Sequential(*perception).eval()
        perception.requires_grad_(False)
        return perception

    @staticmethod
    def _fix_padding(model: nn.Module, padding: PaddingType) -> None:
        """Modify convolution padding mode in the model.

        Args:
            model (nn.Module): The model whose padding to modify.
            padding (PaddingType): Desired padding type.

        Raises:
            ValueError: If padding type is not recognized.

        Note:
            This modifies the model in-place. The changes affect all Conv2d layers
            in the model and its submodules.
        """
        if padding == PaddingType.ZEROS:
            pass  # Default behaviour
        elif padding == PaddingType.REFLECT:
            padding_mode = "reflect"
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    module.padding_mode = padding_mode
        elif padding == PaddingType.VALID:
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    if isinstance(module.padding, (tuple, list)):
                        module.padding = (0,) * len(module.padding)
                    else:
                        module.padding = 0
        else:
            raise ValueError(f"Unknown padding type: {padding}")

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Extract VGG features from input images.

        Args:
            x (torch.Tensor): Input images tensor of shape [B, C, H, W].
                Values should be normalized to range [0, 1].

        Returns:
            Dict[int, torch.Tensor]: Dictionary mapping layer indices to their feature tensors.
                The feature tensors will have varying sizes depending on the layer depth.

        Note:
            1. Input normalization is handled internally using ImageNet statistics.
            2. Features are extracted in eval mode with gradients disabled.
            3. If z_clipping is enabled, features are clipped after each convolution.
        """
        self.perception.eval()
        feats = {}
        x = (x - self.mean) / self.std

        for i, module in enumerate(self.perception):
            x = module(x)
            if isinstance(module, nn.Conv2d):
                x = self.clipper(x)
            if i in self.layers:
                feats[i] = x
        return feats

    def extra_repr(self) -> str:
        return (
            f"Norm: [mean: {self.mean.view(3)}, std: {self.std.view(3)}]\n"
            + f"Layers {self.layers} {[(i, self.perception[i]) for i in self.layers]}\n"
            + f"Z_clip {self.z_clipping}\n"
            + f"Padding {self.padding_type}"
        )


__all__ = ["VGGFeatures", "PaddingType"]

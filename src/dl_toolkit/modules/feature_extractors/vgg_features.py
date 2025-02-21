from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import List, Mapping

import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg19, vgg19_bn, VGG, WeightsEnum, VGG16_Weights, \
    VGG19_Weights, VGG19_BN_Weights

from dl_toolkit.modules.layers.clipper import ClipperChannelwise2D
from dl_toolkit.modules.toolkit_module import ToolkitModule

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class ModelConfig:
    max_layer: int
    weights: WeightsEnum
    func: callable
    mean: List[float] = field(default_factory=lambda: IMAGENET_MEAN)
    std: List[float] = field(default_factory=lambda: IMAGENET_STD)


NETWORKS_CONFIGS: Mapping[str, ModelConfig] = {
    "vgg16": ModelConfig(max_layer=30, weights=VGG16_Weights.IMAGENET1K_V1, func=vgg16),
    "vgg19": ModelConfig(max_layer=36, weights=VGG19_Weights.IMAGENET1K_V1, func=vgg19),
    "vgg19_bn": ModelConfig(max_layer=52, weights=VGG19_BN_Weights.IMAGENET1K_V1, func=vgg19_bn),
}


class PaddingType(Enum):
    ZEROS = 0  # Default VGG padding
    REFLECT = 1  # Replaces zero padding with reflection padding
    VALID = 2  # NO padding, image will be cropped


class VGGFeatures(ToolkitModule):
    VERSION = "1.0.0"

    def __init__(
            self,
            layers: List[int],
            network: str = "vgg16",
            padding_type: PaddingType = PaddingType.ZEROS,
            z_clipping: float | None = None,
    ):
        """VGG features extractor
        :param network: VGG network name
        :param layers: Layers numbers to extract.

        :param z_clipping: apply clipping. Better to use value between [1.5 - 3], will fix some normalizations
        """
        super().__init__()

        assert len(layers) > 0
        assert min(layers) >= 0
        self.layers = list(sorted(set(layers)))

        # Instantiate models
        self.perception = self._get_perception(network, layers)

        # Make normalization of features
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
        constructor = model_config.func
        return constructor(weights=model_config.weights)

    def _get_perception(self, network_name: str, layers: List[int]):
        model_config = NETWORKS_CONFIGS[network_name]
        max_layer = max(layers)
        assert max_layer <= model_config.max_layer, ("Max layer exceeded for "
                                                     f"network {network_name} "
                                                     f"max {self.MAX_LAYER[network_name]}")

        model = self._get_network(model_config)
        perception = list(model.features)[:max_layer + 1]
        perception = nn.Sequential(*perception).eval()
        perception.requires_grad_(False)
        return perception

    @staticmethod
    def _fix_padding(model: nn.Module, padding: PaddingType):
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
                    module.padding = 0
        else:
            raise ValueError(f"Unknown padding type: {padding}")


    def forward(self, x):
        """Calculates VGG features :param x: 4D images tensor.

        Inputs must be normalized in [0..1] range
        :return: list of VGG features defined in layers
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


__all__ = ["VGGFeatures"]

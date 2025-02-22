from typing import Iterable, List

import torch
import torch.nn as nn

from dl_toolkit.modules.feature_extractors.vgg_features import PaddingType, VGGFeatures
from dl_toolkit.modules.losses import CharbonnierLoss
from dl_toolkit.modules.toolkit_module import ToolkitModule

VGG16_LAYERS = [2, 7, 14, 21]
VGG19_LAYERS = [2, 7, 16, 25]
VGG19_BN_LAYERS = [3, 10, 23, 36]


def PerceptualLossSimple(
    model_name: str = "vgg19_bn",
    loss_type: str = "charbonnier",
    z_clip: float | None = None,
    fix_pad=False,
    apply_norm=True,
    use_last_layers: int | None = None,
):
    """
    Perceptual loss for image generation. Simplified version with good parameters
    """
    name2layers = {"vgg16": VGG16_LAYERS, "vgg19": VGG19_LAYERS, "vgg19_bn": VGG19_BN_LAYERS}
    layers = name2layers[model_name]
    if use_last_layers:
        assert len(layers) >= use_last_layers > 0
        layers = layers[-use_last_layers:]
    padding = PaddingType.REFLECT if fix_pad else PaddingType.ZEROS
    return PerceptualLoss(
        model_name=model_name,
        layers=layers,
        apply_norm=apply_norm,
        padding=padding,
        loss_type=loss_type,
        z_clip=z_clip,
    )


class PerceptualLoss(ToolkitModule):
    VERSION = "1.0.0"

    # Layers from https://towardsdatascience.com/implementing-neural-style-transfer-using-pytorch-fd8d43fb7bfa
    def __init__(
        self,
        model_name: str = "vgg19",
        layers: List[int] = (),
        apply_norm: bool = False,
        loss_type: str = "smooth_l1",
        weight_scaler: float = 2,
        reverse_weights: bool = False,
        padding: PaddingType = PaddingType.ZEROS,
        z_clip: float | None = None,
    ):
        super().__init__()
        assert (
            layers is not None and len(layers) > 0
        ), "Please add layers to content loss. i.e. [25] for vgg19 or [36] for vgg19"

        self.apply_norm = apply_norm
        assert apply_norm in [True, False, "both"]
        assert loss_type in ["l1", "smooth_l1", "l2", "charbonnier"]
        self.base_loss = self.get_loss(loss_type)
        self.padding = padding
        self.z_clip = z_clip
        self.feature_extractor = self.get_model(model_name=model_name, layers=layers)

        weights = self.get_weights(len(layers), weight_scaler, reversed_weight=reverse_weights)
        self.layers = dict(zip(list(layers), weights))
        self.norm = self.get_norm(512)  # Instance Norm ignores num_channels

    def get_weights(self, num_layers, weight_scaler, reversed_weight) -> List[float]:
        weights = list(
            reversed([1 / (weight_scaler**i) for i in range(num_layers)])
        )  # Large weight at the end
        if reversed_weight:
            weights = list(reversed(weights))
        sum_weight = sum(weights)
        weights = [weight / sum_weight for weight in weights]
        return weights

    def get_model(self, model_name, layers=()):
        return VGGFeatures(
            network=model_name, layers=layers, padding_type=self.padding, z_clipping=self.z_clip
        )

    def get_loss(self, loss_type):
        loss_type = loss_type.lower()
        if loss_type == "l1":
            return nn.L1Loss()
        elif loss_type == "smooth_l1":
            beta = 0.2
            return nn.SmoothL1Loss(beta=beta)
        elif loss_type == "l2":
            return nn.MSELoss()
        elif loss_type == "charbonnier":
            return CharbonnierLoss()
        else:
            raise ValueError(f"Error with loss type {loss_type}")

    def get_norm(self, num_channels):
        return nn.InstanceNorm2d(
            num_channels, affine=False, track_running_stats=False
        )  # Please don't touch eps in norm, it affects image gray content

    def forward(self, pred, target):
        self.norm.train()
        pred = self.feature_extractor(pred)
        with torch.no_grad():
            target = self.feature_extractor(target)
        loss = 0
        for layer, weight in self.layers.items():
            pred_i = pred[layer]
            target_i = target[layer]
            if self.apply_norm:
                pred_i = self.norm(pred_i)
                target_i = self.norm(target_i)
            loss += self.base_loss(pred_i, target_i) * weight
        return loss

    def extra_repr(self) -> str:
        return (
            f"Layers_weights: {self.layers}\n"
            f"Apply norm: {self.apply_norm}\n"
            f"Feature extractor: {self.feature_extractor}"
        )

from typing import Tuple

import torch
import torch.nn as nn

from dl_toolkit.modules.losses.utils.reduction import reduce_data
from dl_toolkit.modules.toolkit_module import ToolkitModule
from dl_toolkit.modules.utils.math import logit


class GANLoss(ToolkitModule):
    VERSION = "1.0.1"

    def __init__(
            self, criterion: nn.Module = nn.BCEWithLogitsLoss(), is_logit: bool = True,
            clip: float | None = None
    ):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))
        self.use_clip = clip is not None and clip > 0
        if self.use_clip:
            if clip >= 1.0 or clip <= 0.0:
                raise ValueError("clip must be in range (0.0, 1.0)")
            self.clip_min, self.clip_max = self.__init_clip(clip, is_logit)
        self.is_logit = is_logit
        self.base_loss = criterion
        if self.is_logit and isinstance(self.base_loss, nn.BCELoss):
            raise ValueError("BCELoss should be used with logit=False")
        if not self.is_logit and isinstance(self.base_loss, (nn.BCEWithLogitsLoss)):
            raise ValueError("BCEWithLogitsLoss should be used with logit=True")

    def forward(self, pred, target_is_real):
        if self.use_clip:
            pred = self.clip_tensor(pred)
        return self.base_loss(pred, self.get_target_tensor(pred, target_is_real))

    @staticmethod
    def __init_clip(clip: float, is_logit: bool) -> Tuple[float, float]:
        if clip >= 0.5:
            raise ValueError("clip must be less than 0.5")
        clip_min = clip
        clip_max = 1 - clip
        if is_logit:
            clip_min = logit(clip_min)
            clip_max = logit(clip_max)
        return clip_min, clip_max

    def clip_tensor(self, pred):
        return torch.clip(pred, min=self.clip_min, max=self.clip_max)

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def extra_repr(self):
        return f"is_logit={self.is_logit}, clip=({self.clip_min:.2f}, {self.clip_max:.2f})"


class SoftplusGANLoss(GANLoss):
    VERSION = "1.0.1"

    """
    Loss function for Generative Adversarial Networks proposed by Goodfellow et al (2014)
    Look at https://arxiv.org/pdf/1406.2661 Section 3, train to maximize log(D(G(z))). This version
    of loss used in current implementation of StyleGAN2.
    """

    #
    def __init__(self, is_logit=True, clip: float | None = None, reduction="mean"):
        if not is_logit:
            raise ValueError("SoftPlusGANLoss should be used with logit=True")
        self.reduction = reduction
        super().__init__(nn.Softplus(), is_logit=is_logit, clip=clip)

    def forward(self, pred, target_is_real):
        if self.use_clip:
            pred = self.clip_tensor(pred)
        multiplier = -1.0 if target_is_real else 1.0
        return reduce_data(self.base_loss(multiplier * pred), reduction=self.reduction)

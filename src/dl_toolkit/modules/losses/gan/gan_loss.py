from typing import Tuple

import torch
import torch.nn as nn

from dl_toolkit.modules.toolkit_module import ToolkitModule
from dl_toolkit.modules.utils.math import logit


class GANLoss(ToolkitModule):
    def __init__(
        self, criterion: nn.Module = nn.BCELoss(), is_logit: bool = True, clip: float | None = None
    ):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))
        self.use_clip = clip is not None and clip > 0
        if self.use_clip:
            self.clip_min, self.clip_max = self.__init_clip(clip, is_logit)
        self.is_logit = is_logit
        self.base_loss = criterion

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

from collections.abc import Sequence, Mapping
from typing import Iterable

import torch
import torch.nn as nn

from dl_toolkit.modules.toolkit_module import ToolkitModule
from dl_toolkit.utils.interpolation.interpolator import AbstractInterpolator


class LossWrapper(ToolkitModule):
    VERSION = "1.0.0"

    def __init__(
            self,
            loss: nn.Module,
            weight: float = 1.0,
            interpolator: AbstractInterpolator | None = None,
            loss_output_ids: Iterable[int | str] = (0,),
    ):
        """
        Loss wrapper to make it more stable and generic for training.

        Args:
            loss (nn.Module): The loss module to wrap.
            weight (float): Weight to multiply the loss value.
            interpolator: (AbstractInterpolator): Interpolator to warmup or decay the weight.
                Might be useful in case when loss have huge impact on training or don't need
                at the end of training.
            loss_output_ids: Iterable(int, str): In cases where the loss has multiple outputs,
                we multiply those ids or keys by weight.
        """
        super().__init__()
        if weight <= 0:
            self.loss = lambda *args, **kwargs: 0
        else:
            self.loss = loss
        self.weight = weight

        self.interpolator = None
        self.register_buffer("num_steps", torch.zeros([], dtype=torch.long))
        self.register_buffer("zero_loss", torch.zeros([], dtype=torch.float32))
        self.interpolator = interpolator
        self.loss_output_ids = loss_output_ids

    def forward(self, *args, **kwargs):
        warmup_weight = 1
        if self.training and self.interpolator is not None:
            num_steps = self.num_steps.item()
            warmup_weight = self.interpolator(num_steps)
            self.num_steps += 1
        if self.weight > 0 and warmup_weight > 0:
            loss = self.loss(*args, **kwargs)
            multiplier = self.weight * warmup_weight
            if isinstance(loss, (tuple, Sequence, Mapping)):
                # Make editable copy
                if isinstance(loss, Mapping):
                    outputs = dict(**loss)
                elif isinstance(loss, Sequence):
                    outputs = list(loss)
                else:
                    raise TypeError("Loss is not a tuple, sequence or mapping.")

                for mul_id in self.loss_output_ids:
                    outputs[mul_id] = outputs[mul_id] * multiplier
                return type(loss)(outputs)  # Return same type as loss
            else:
                return loss * multiplier
        else:
            return self.zero_loss

    def denorm_loss(self, loss_value):
        if self.weight > 0:
            return loss_value / self.weight
        else:
            return self.zero_loss

    def reset(self):
        self.num_steps.zero_()

    def extra_repr(self) -> str:
        if self.interpolator is not None:
            return f"weight={self.weight}, interpolator={self.interpolator}"
        else:
            return f"weight={self.weight}"

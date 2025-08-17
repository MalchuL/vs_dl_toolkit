from collections.abc import Sequence, Mapping
from typing import Iterable, Optional

import torch
import torch.nn as nn

from dl_toolkit.modules.toolkit_module import ToolkitModule
from dl_toolkit.utils.interpolation.interpolator import AbstractInterpolator


class LossWrapper(ToolkitModule):
    VERSION = "2.0.0"

    def __init__(
            self,
            loss: nn.Module,
            weight: float = 1.0,
            weight_min: Optional[float] = None,
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
            weight_min (float): Minimal weight of loss, uses to train with small weight in warmup
            loss_output_ids: Iterable(int, str): In cases where the loss has multiple outputs,
                we multiply those ids or keys by weight.
        """
        super().__init__()
        if weight <= 0:
            self.loss = lambda *args, **kwargs: 0
        else:
            self.loss = loss
        self.weight = weight
        self.weight_min = weight_min
        if self.weight_min is not None:
            if interpolator is None:
                raise ValueError("interpolator must be provided if weight_min is not None")
            if self.weight_min > self.weight:
                raise ValueError("weight_min must be less than or equal to weight, "
                                 "got {self.weight_min} > {self.weight}")

        self.interpolator = None
        self.register_buffer("_num_steps", torch.zeros([], dtype=torch.long))
        self.register_buffer("_zero_loss", torch.zeros([], dtype=torch.float32))
        self.interpolator = interpolator
        self.loss_output_ids = loss_output_ids

    @property
    def current_weight(self) -> float:
        warmup_weight = 1
        if self.interpolator is not None:
            num_steps = self.num_steps
            # Warmup weight can be between 0 and 1
            warmup_weight = self.interpolator(num_steps)
        # In case of eval we don't use warmup
        if not self.training:
            warmup_weight = 1
        weight_min = 0
        if self.weight_min is not None:
            weight_min = self.weight_min

        if self.weight > 0:
            return (self.weight - weight_min) * warmup_weight + weight_min
        else:
            # We don't use warmup_weight in case of weight=0
            return 0

    def forward(self, *args, **kwargs):
        if self.training:
            self.num_steps += 1
        if self.current_weight > 0:
            loss = self.loss(*args, **kwargs)
            multiplier = self.current_weight
            if isinstance(loss, (Sequence, Mapping)):
                # Make editable copy
                if isinstance(loss, Mapping):
                    outputs = dict(**loss)
                elif isinstance(loss, Sequence):
                    outputs = list(loss)
                else:
                    raise TypeError("Loss is not a tuple, sequence or mapping.")

                for mul_id in self.loss_output_ids:
                    outputs[mul_id] = outputs[mul_id] * multiplier
                return type(loss)(outputs)  # Return same type as loss, dict or list
            else:
                return loss * multiplier
        else:
            return self._zero_loss

    def denorm_loss(self, loss_value):
        if self.current_weight > 0:
            return loss_value / self.current_weight
        else:
            return self._zero_loss

    def is_always_zero(self):
        return self.weight == 0

    @property
    def num_steps(self) -> int:
        return self._num_steps.item()

    @num_steps.setter
    def num_steps(self, value: int):
        self._num_steps.fill_(value)

    def reset(self):
        self._num_steps.fill_(0)

    def extra_repr(self) -> str:
        if self.interpolator is not None:
            return (f"weight={self.weight}, interpolator={self.interpolator}, "
                    "weight_min={self.weight_min}")
        else:
            return f"weight={self.weight}"

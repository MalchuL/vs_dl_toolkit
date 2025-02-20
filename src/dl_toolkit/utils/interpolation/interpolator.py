from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple

from dl_toolkit.utils.logging import logger
from .tweenings import interpolate
from enum import Enum


class InterpolationMode(Enum):
    """
    Interpolation modes. You can look at https://pypi.org/project/pytweening/
    """
    LINEAR = "linear"
    EASE_IN_QUAD = "easeInQuad"
    EASE_OUT_QUAD = "easeOutQuad"
    EASE_IN_OUT_QUAD = "easeInOutQuad"
    EASE_IN_CUBIC = "easeInCubic"
    EASE_OUT_CUBIC = "easeOutCubic"
    EASE_IN_OUT_CUBIC = "easeInOutCubic"
    EASE_IN_QUART = "easeInQuart"
    EASE_OUT_QUART = "easeOutQuart"
    EASE_IN_OUT_QUART = "easeInOutQuart"
    EASE_IN_QUINT = "easeInQuint"
    EASE_OUT_QUINT = "easeOutQuint"
    EASE_IN_OUT_QUINT = "easeInOutQuint"
    EASE_IN_SINE = "easeInSine"
    EASE_OUT_SINE = "easeOutSine"
    EASE_IN_OUT_SINE = "easeInOutSine"
    EASE_IN_EXPO = "easeInExpo"
    EASE_OUT_EXPO = "easeOutExpo"
    EASE_IN_OUT_EXPO = "easeInOutExpo"
    EASE_IN_CIRC = "easeInCirc"
    EASE_OUT_CIRC = "easeOutCirc"
    EASE_IN_OUT_CIRC = "easeInOutCirc"
    EASE_OUT_ELASTIC = "easeOutElastic"
    EASE_OUT_BACK = "easeOutBack"
    EASE_IN_BOUNCE = "easeInBounce"
    EASE_OUT_BOUNCE = "easeOutBounce"
    EASE_IN_OUT_BOUNCE = "easeInOutBounce"
    EASE_IN_POLY = "easeInPoly"
    EASE_OUT_POLY = "easeOutPoly"
    EASE_IN_OUT_POLY = "easeInOutPoly"


class Direction(Enum):
    UP = "up"  # Start from 0.0 and end at 1.0
    DOWN = "down"  # Start from 1.0 and end at 0.0
    CONSTANT_0 = "constant_0"  # Always 0.0
    CONSTANT_1 = "constant_1"  # Always 1.0


class AbstractInterpolator(ABC):

    def __init__(self, num_steps):
        if num_steps <= 0:
            raise ValueError("num_steps must be greater than 0")
        self.num_steps = num_steps

    @abstractmethod
    def interpolate(self, step: int) -> float:
        """
        Returns interpolated value between 0.0 and 1.0 based on num_steps. If step is out of range,
        returns 0.0 or 1.0.
        Args:
            step (int): current step
        """

    def alpha_interpolate(self, alpha: float) -> float:
        return self.alpha_interpolate(round(alpha * self.num_steps))

    def __call__(self, step):
        return self.interpolate(step)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_steps}"


class Interpolator(AbstractInterpolator):
    def __init__(self, num_steps: int,
                 method: str | InterpolationMode | None = InterpolationMode.LINEAR,
                 direction: str | Direction = Direction.UP):
        """
        method - string name of pytweening lib
        num_steps - maximal number of steps
        """
        super().__init__(num_steps)
        if isinstance(method, str):
            method = InterpolationMode(method)
        self.method = method

        if isinstance(direction, str):
            direction = Direction(direction)
        self.direction = direction

        self.__check_method(self.method, self.direction, self.num_steps)

    def interpolate(self, step: int):
        # Constant cases
        if self.direction == Direction.CONSTANT_0:
            return 0.0
        elif self.direction == Direction.CONSTANT_1:
            return 1.0
        else:
            alpha = step / self.num_steps
            out_alpha = interpolate(alpha, self.method.value)
            if self.direction == Direction.DOWN:
                # Important: we reverse output instead of reverse alpha, because
                #            tweening is not symmetric
                out_alpha = 1 - out_alpha
        return out_alpha

    @staticmethod
    def __check_method(method: InterpolationMode | None, direction: Direction, num_steps: int):
        if direction in (Direction.CONSTANT_0, Direction.CONSTANT_1):
            if method is not None:
                raise ValueError("Method is not applicable for constant direction, "
                                 "pass method=None")
        else:
            if method is None:
                raise ValueError(f"Method is required to be not None "
                                 "for dynamic direction={direction}")
            if num_steps <= 0:
                raise ValueError("num_steps must be greater than 0 for dynamic direction, choose "
                                 "CONSTANT_0 or CONSTANT_1")

        # Specific interpolation methods will be checked here
        if method in [InterpolationMode.EASE_OUT_ELASTIC, InterpolationMode.EASE_OUT_BACK]:
            logger.warning(f"Method {method} used for interpolation would be larger "
                           "that 1 (1.4 at max)")
            if direction == Direction.DOWN:
                raise ValueError(f"Method {method} can't be used for direction {direction}, "
                                 "because it would be larger that 1 (1.4 at max)")
        elif method in [InterpolationMode.EASE_IN_BOUNCE, InterpolationMode.EASE_IN_OUT_BOUNCE]:
            logger.warning(f"Method {method} might be little bit lower "
                           "than 0 (-0.01 at min), please note that")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_steps}, method={self.method}, " \
               f"direction={self.direction})"


class MultiInterpolator(AbstractInterpolator):
    def __init__(self, interpolators: List[Interpolator]):
        num_steps = sum([interpolator.num_steps for interpolator in interpolators])
        super().__init__(num_steps)
        self.interpolators = interpolators
        self.milestones = []
        current_milestone = 0
        for interpolator in self.interpolators:
            current_milestone += interpolator.num_steps
            self.milestones.append({"milestone": current_milestone, "interpolator": interpolator})

    def interpolate(self, step):
        prev_milestone = 0
        for interpolation in self.milestones[:-1]:
            interpolator = interpolation["interpolator"]
            milestone = interpolation["milestone"]
            if step < milestone:
                return interpolator.interpolate(step - prev_milestone)
            prev_milestone = milestone

        return self.milestones[-1]["interpolator"].interpolate(step - prev_milestone)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_steps}, interpolators={self.interpolators})"
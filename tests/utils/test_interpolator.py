import numpy as np
import pytest
from dl_toolkit.utils.interpolation import InterpolationMode, \
    Interpolator, Direction, interpolate, MultiInterpolator

from dl_toolkit.utils.logging import logger

NUM_STEPS = 1000


def test_tweenings():
    steps = list(range(NUM_STEPS + 1))
    alphas = [a / NUM_STEPS for a in steps]
    for mode in InterpolationMode:
        logger.info(f"{mode} interpolation")
        for alpha in alphas:
            inter_value = interpolate(alpha=alpha, method=mode.value)
            if mode in [InterpolationMode.EASE_OUT_ELASTIC, InterpolationMode.EASE_OUT_BACK]:
                assert 0.0 <= inter_value <= 1.5, f"{inter_value} is out of range for mode {mode}"
            elif mode in [InterpolationMode.EASE_IN_BOUNCE, InterpolationMode.EASE_IN_OUT_BOUNCE]:
                assert -1e-2 <= inter_value <= 1 + 1e-2, f"{inter_value} is out of range for mode {mode}"
            else:
                assert 0.0 <= inter_value <= 1.01, f"{inter_value} is out of range for mode {mode}"


def test_wrong_interpolators():
    with pytest.raises(ValueError):
        interpolator = Interpolator(0)  # You can't create interpolator with zero steps
    with pytest.raises(ValueError):
        interpolator = Interpolator(100, method=InterpolationMode.EASE_IN_BOUNCE,
                                    direction=Direction.CONSTANT_1)  # You can't create interpolator with constant direction and interpolation method other than None
    with pytest.raises(ValueError):
        interpolator = Interpolator(100, method=InterpolationMode.EASE_OUT_BACK,
                                    direction=Direction.DOWN)  # You can't create interpolator with thoose parameters
    with pytest.raises(ValueError):
        interpolator = Interpolator(2000, method=None, direction=Direction.DOWN)


def test_constant_interpolator():
    steps = list(range(NUM_STEPS + 1))
    steps.extend([-1, -2, -3, -4, -5, NUM_STEPS, NUM_STEPS + 1, NUM_STEPS + 2, NUM_STEPS + 3])
    interpolator_1 = Interpolator(NUM_STEPS, method=None, direction=Direction.CONSTANT_1)
    interpolator_0 = Interpolator(NUM_STEPS, method=None, direction=Direction.CONSTANT_0)
    for step in steps:
        inter_value_1 = interpolator_1(step)
        inter_value_0 = interpolator_0(step)
        assert inter_value_1 == 1.0
        assert inter_value_0 == 0.0


@pytest.mark.parametrize("method", [InterpolationMode.LINEAR, InterpolationMode.EASE_OUT_CIRC, InterpolationMode.EASE_OUT_EXPO])
@pytest.mark.parametrize("direction", [Direction.UP, Direction.DOWN])
def test_interpolation_constructor(method, direction):
    steps = [-2, -1, 0, 1, 2, 3, 5, 6, 7]
    num_steps = 5
    interpolator = Interpolator(num_steps, method=method, direction=direction)
    alphas = np.array([interpolator(step) for step in steps])
    diffs = alphas[1:] - alphas[:-1]
    if direction == Direction.UP:
        assert np.all(diffs >= 0)
        assert np.all(alphas[:3] == 0.0)
        assert np.all(alphas[-2:] == 1.0)
    else:
        assert np.all(diffs <= 0)
        assert np.all(alphas[:3] == 1.0)
        assert np.all(alphas[-2:] == 0.0)


def test_multiple_interpolators():
    interpolator1 = Interpolator(1000, direction=Direction.UP)
    interpolator2 = Interpolator(5000, method=None, direction=Direction.CONSTANT_1)
    interpolator3 = Interpolator(2000, method=InterpolationMode.EASE_IN_CIRC,
                                 direction=Direction.DOWN)
    interpolator = MultiInterpolator([interpolator1, interpolator2, interpolator3])
    steps = 10000
    x = list(range(steps))
    y = [interpolator(el) for el in x]
    assert y[0] == 0.0
    assert y[9000] == 0.0
    assert y[4000] == 1.0
    assert 0.45 < y[500] < 0.55
    assert 0.95 < y[6500] < 0.98
    assert 0.86 < y[7000] < 0.87


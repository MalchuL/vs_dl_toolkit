import pytest
import torch
import torch.nn as nn

from dl_toolkit.modules.losses.loss_wrapper import LossWrapper
from dl_toolkit.utils.interpolation import (
    Direction,
    InterpolationMode,
    Interpolator,
    MultiInterpolator,
)


# Mock Loss Modules
class MockSingleOutputLoss(nn.Module):
    def forward(self, *args, **kwargs):
        return torch.tensor(1.0)


class MockMultiOutputLoss(nn.Module):
    def forward(self, *args, **kwargs):
        return (torch.tensor(1.0), torch.tensor(2.0))


class MockMultiOutputDictLoss(nn.Module):
    def forward(self, *args, **kwargs):
        return {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}


# Fixtures
@pytest.fixture
def single_output_loss():
    return MockSingleOutputLoss()


@pytest.fixture
def multi_output_loss():
    return MockMultiOutputLoss()


@pytest.fixture
def multi_output_dict_loss():
    return MockMultiOutputDictLoss()


@pytest.fixture
def linear_interpolator():
    return Interpolator(num_steps=10, method=InterpolationMode.LINEAR, direction=Direction.UP)


@pytest.fixture
def constant_interpolator():
    return Interpolator(num_steps=10, method=None, direction=Direction.CONSTANT_1)


@pytest.fixture
def multi_interpolator():
    interpolators = [
        Interpolator(num_steps=5, method=InterpolationMode.LINEAR, direction=Direction.UP),
        Interpolator(num_steps=5, method=InterpolationMode.LINEAR, direction=Direction.DOWN),
    ]
    return MultiInterpolator(interpolators)


# Tests
def test_basic_weight_scaling(single_output_loss):
    wrapper = LossWrapper(loss=single_output_loss, weight=2.0)
    loss_value = wrapper()
    assert torch.is_tensor(loss_value)
    assert loss_value.item() == pytest.approx(2.0)


def test_zero_weight_disables_loss(single_output_loss):
    wrapper = LossWrapper(loss=single_output_loss, weight=0.0)
    loss_value = wrapper()
    assert loss_value.item() == 0  # Returns scalar 0


def test_negative_weight_disables_loss(single_output_loss):
    wrapper = LossWrapper(loss=single_output_loss, weight=-1.0)
    loss_value = wrapper()
    assert loss_value.item() == 0  # Returns scalar 0


def test_interpolator_application(single_output_loss, linear_interpolator):
    wrapper = LossWrapper(loss=single_output_loss, weight=2.0, interpolator=linear_interpolator)
    wrapper.train()

    # Step 1
    for _ in range(2):
        loss_value = wrapper()
    expected = 2.0 * (1 / 10)  # Linear interpolation at step 1
    assert loss_value == pytest.approx(expected)
    assert wrapper.num_steps.item() == 2

    # Step 10
    for _ in range(8):
        wrapper()
    assert wrapper.num_steps.item() == 10
    loss_value = wrapper()
    expected = 2.0 * 1.0  # Linear interpolation at step 10
    assert loss_value.item() == pytest.approx(expected)


def test_multi_output_scaling(multi_output_loss):
    wrapper = LossWrapper(loss=multi_output_loss, weight=2.0, loss_output_ids=(0, 1))
    loss_values = wrapper()

    assert isinstance(loss_values, tuple)
    assert loss_values[0].item() == pytest.approx(2.0)  # 1.0 * 2.0
    assert loss_values[1].item() == pytest.approx(4.0)  # 2.0 * 2.0


def test_multi_output_dict_scaling(multi_output_dict_loss):
    wrapper = LossWrapper(loss=multi_output_dict_loss, weight=2.0, loss_output_ids=("a", "b"))
    loss_values = wrapper()

    assert isinstance(loss_values, dict)
    assert loss_values["a"].item() == pytest.approx(2.0)  # 1.0 * 2.0
    assert loss_values["b"].item() == pytest.approx(4.0)  # 2.0 * 2.0


def test_denormalization(single_output_loss):
    wrapper = LossWrapper(loss=single_output_loss, weight=2.0)
    loss_value = wrapper()
    denorm_value = wrapper.denorm_loss(loss_value)
    assert denorm_value.item() == pytest.approx(1.0)


def test_extra_repr(single_output_loss, linear_interpolator):
    wrapper = LossWrapper(loss=single_output_loss, weight=2.0, interpolator=linear_interpolator)
    repr_str = wrapper.extra_repr()
    assert "weight=2.0" in repr_str
    assert "interpolator" in repr_str


def test_eval_mode_behavior(single_output_loss, linear_interpolator):
    wrapper = LossWrapper(loss=single_output_loss, weight=2.0, interpolator=linear_interpolator)
    wrapper.eval()  # Disable training mode

    loss_value = wrapper()
    assert loss_value.item() == pytest.approx(2.0)  # No interpolation
    assert wrapper.num_steps == 0  # Step counter not incremented


def test_constant_interpolator(single_output_loss, constant_interpolator):
    wrapper = LossWrapper(loss=single_output_loss, weight=2.0, interpolator=constant_interpolator)
    wrapper.train()

    loss_value = wrapper()
    assert loss_value == pytest.approx(2.0)  # Constant 1.0 * 2.0


def test_multi_interpolator(single_output_loss, multi_interpolator):
    wrapper = LossWrapper(loss=single_output_loss, weight=2.0, interpolator=multi_interpolator)
    wrapper.train()

    # Step 1 (first interpolator)
    for _ in range(2):
        loss_value = wrapper()
    expected = 2.0 * (1 / 5)  # Linear interpolation at step 1
    assert loss_value == pytest.approx(expected)

    # Step 6 (second interpolator)
    for _ in range(5):
        wrapper()
    loss_value = wrapper()
    expected = 2.0 * (3 / 5)  # Linear interpolation at step 6
    assert loss_value == pytest.approx(expected)


def test_invalid_interpolator_method():
    with pytest.raises(ValueError):
        Interpolator(
            num_steps=10, method=InterpolationMode.EASE_OUT_ELASTIC, direction=Direction.DOWN
        )


def test_invalid_interpolator_direction():
    with pytest.raises(ValueError):
        Interpolator(num_steps=10, method=None, direction=Direction.UP)


def test_dtype_for_zero(single_output_loss):
    wrapper = LossWrapper(loss=single_output_loss, weight=0)
    wrapper.train()
    wrapper.to(torch.float16)
    loss_value = wrapper()
    assert wrapper.num_steps.dtype == torch.long  # Step counter should be long
    assert loss_value.dtype == torch.float16
    assert loss_value == pytest.approx(0)  # Constant 0

import pytest
import torch
from torch import nn

from dl_toolkit.modules.losses.gan.rp_gan_loss import RPGANLoss


def test_initialization():
    """Test default initialization parameters."""
    loss = RPGANLoss()
    assert isinstance(loss.criterion, nn.Softplus)
    assert loss.is_logit is True
    assert loss.reduction == "mean"


def test_forward_generator_loss():
    """Test forward pass for generator loss (is_generator_loss=True)."""
    loss = RPGANLoss(reduction="none")
    fake_pred = torch.tensor([1.0])
    real_pred = torch.tensor([0.5])

    output = loss(fake_pred, real_pred, is_generator_loss=True)
    expected = nn.functional.softplus(torch.tensor(-0.5))  # softplus(-(1.0 - 0.5))

    assert torch.allclose(output, expected)


def test_forward_discriminator_loss():
    """Test forward pass for discriminator loss (is_generator_loss=False)."""
    loss = RPGANLoss(reduction="none")
    fake_pred = torch.tensor([0.2])
    real_pred = torch.tensor([0.8])

    output = loss(fake_pred, real_pred, is_generator_loss=False)
    expected = nn.functional.softplus(torch.tensor(-0.6))  # softplus(-(0.8 - 0.2))

    assert torch.allclose(output, expected)


def test_logit_conversion():
    """Test logit conversion when is_logit=False."""
    loss = RPGANLoss(is_logit=False)
    fake_pred = torch.sigmoid(torch.tensor([2.0]))
    real_pred = torch.sigmoid(torch.tensor([1.0]))

    # Convert back using logit and compute loss
    output = loss(fake_pred, real_pred, is_generator_loss=True)
    expected = nn.functional.softplus(torch.tensor(-(2.0 - 1.0)))

    assert torch.allclose(output, expected)


@pytest.mark.parametrize("reduction, expected_shape", [
    ("none", (2, 3)),
    ("mean", ()),
    ("sum", ())
])
def test_reduction_methods(reduction, expected_shape):
    """Test different reduction methods."""
    loss = RPGANLoss(reduction=reduction)
    fake_pred = torch.randn(2, 3)
    real_pred = torch.randn(2, 3)
    output = loss(fake_pred, real_pred, is_generator_loss=True)
    assert output.shape == expected_shape


def test_batch_processing():
    """Test processing of batch inputs."""
    loss = RPGANLoss(reduction="mean")
    fake_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    real_pred = torch.tensor([[0.5, 1.5], [2.5, 3.5]])

    output = loss(fake_pred, real_pred, is_generator_loss=True)
    relativistic = fake_pred - real_pred
    expected = nn.functional.softplus(-relativistic).mean()

    assert torch.allclose(output, expected)


def test_dtype_preservation():
    """Test preservation of input dtype."""
    loss = RPGANLoss()
    fake_pred = torch.randn(2, 3, dtype=torch.float64)
    real_pred = torch.randn(2, 3, dtype=torch.float64)
    output = loss(fake_pred, real_pred, is_generator_loss=True)
    assert output.dtype == torch.float64


def test_average_relative_loss():
    loss = RPGANLoss(rel_avg_gan=True)
    high_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    low_pred = torch.tensor([[0.5, 1.5], [2.5, 3.5]])

    # For generator loss, high fake pred is good for generator
    output_good_generator = loss(high_pred, low_pred, is_generator_loss=True)
    output_bad_generator = loss(low_pred, high_pred, is_generator_loss=True)
    assert output_good_generator.shape == ()
    assert output_good_generator < output_bad_generator

    # For discriminator loss high fake pred is bad for discriminator
    output_good_generator = loss(high_pred, low_pred, is_generator_loss=False)
    output_bad_generator = loss(low_pred, high_pred, is_generator_loss=False)
    assert output_good_generator.shape == ()
    assert output_good_generator > output_bad_generator
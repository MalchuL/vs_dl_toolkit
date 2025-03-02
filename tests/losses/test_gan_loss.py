import pytest
import torch
from torch.nn import BCELoss, BCEWithLogitsLoss

from dl_toolkit.modules.losses.gan.gan_loss import GANLoss
from dl_toolkit.modules.utils.math import logit


@pytest.fixture
def input_tensor():
    return torch.randn(2, 3, 4)


# Test initialization of GANLoss
@pytest.mark.parametrize("clip, is_logit, expected_min, expected_max", [
    (0.2, False, 0.2, 0.8),
    (0.3, True, logit(torch.tensor(0.3)).item(), logit(torch.tensor(0.7)).item()),
])
def test_gan_loss_init(clip, is_logit, expected_min, expected_max):
    gan_loss = GANLoss(clip=clip, is_logit=is_logit)
    assert gan_loss.use_clip
    assert gan_loss.clip_min == pytest.approx(expected_min, abs=1e-4)
    assert gan_loss.clip_max == pytest.approx(expected_max, abs=1e-4)


def test_gan_loss_init_invalid_clip():
    with pytest.raises(ValueError):
        GANLoss(clip=0.5)


# Test target tensor generation
@pytest.mark.parametrize("target_is_real", [True, False])
def test_get_target_tensor(input_tensor, target_is_real):
    gan_loss = GANLoss()
    target = gan_loss.get_target_tensor(input_tensor, target_is_real)
    expected_value = 1.0 if target_is_real else 0.0
    assert torch.all(target == expected_value)
    assert target.shape == input_tensor.shape


# Test tensor clipping functionality
@pytest.mark.parametrize("is_logit, clip, input_vals, expected_vals", [
    (False, 0.2, [0.1, 0.3, 0.9], [0.2, 0.3, 0.8]),
    (True, 0.1, [logit(0.05), logit(0.5), logit(0.95)],
     [logit(0.1), logit(0.5), logit(0.9)]),
])
def test_clip_tensor(is_logit, clip, input_vals, expected_vals):
    gan_loss = GANLoss(clip=clip, is_logit=is_logit)
    input_tensor = torch.tensor(input_vals)
    clipped = gan_loss.clip_tensor(input_tensor)
    expected = torch.tensor(expected_vals)
    assert torch.allclose(clipped, expected, atol=1e-4)


# Test forward pass with different configurations
@pytest.mark.parametrize("criterion, is_logit, clip, pred, target_is_real, expected", [
    (BCELoss(), False, None, [1.0], True, 0.0),
    (BCELoss(), False, 0.1, [0.0], True, BCELoss()(torch.tensor([0.1]), torch.tensor([1.0]))),
    (BCEWithLogitsLoss(), True, 0.2, [2.0], True,
     BCEWithLogitsLoss()(torch.tensor([logit(0.8)]), torch.tensor([1.0]))),
    (BCELoss(), False, None, [0.4], False, BCELoss()(torch.tensor([0.4]), torch.tensor([0.0]))),
])
def test_forward_pass(criterion, is_logit, clip, pred, target_is_real, expected):
    gan_loss = GANLoss(criterion=criterion, is_logit=is_logit, clip=clip)
    pred_tensor = torch.tensor(pred, dtype=torch.float32)
    loss = gan_loss(pred_tensor, target_is_real)
    expected_loss = expected if isinstance(expected, float) else expected.item()
    assert loss.item() == pytest.approx(expected_loss, abs=1e-4)


# Test string representation
def test_extra_repr():
    # Test with clipping
    gan_loss = GANLoss(clip=0.2, is_logit=False)
    assert "is_logit=False" in gan_loss.extra_repr()
    assert "clip=(0.20, 0.80)" in gan_loss.extra_repr()

    # Test with logit clipping
    gan_loss_logit = GANLoss(clip=0.3, is_logit=True)
    clip_min = logit(torch.tensor(0.3)).item()
    clip_max = logit(torch.tensor(0.7)).item()
    assert f"clip=({clip_min:.2f}, {clip_max:.2f})" in gan_loss_logit.extra_repr()
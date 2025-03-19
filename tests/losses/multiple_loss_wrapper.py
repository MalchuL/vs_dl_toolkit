import pytest
import torch
import torch.nn as nn

from dl_toolkit.modules.losses.merging_loss import MergingLossWrapper


class MockLoss(nn.Module):
    """A mock loss module that returns a fixed value."""
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, *args, **kwargs):
        return torch.tensor(self.value, dtype=torch.float32)

def test_default_weights():
    """Test that all losses default to a weight of 1.0."""
    losses = {
        'loss1': MockLoss(2.0),
        'loss2': MockLoss(3.0),
    }
    merger = MergingLossWrapper(losses)
    total_loss = merger()
    assert torch.isclose(total_loss, torch.tensor(5.0))

def test_custom_weights():
    """Test that losses are scaled by their custom weights."""
    losses = {
        'loss1': MockLoss(2.0),
        'loss2': MockLoss(3.0),
    }
    weights = {'loss1': 0.5, 'loss2': 2.0}
    merger = MergingLossWrapper(losses, weights)
    total_loss = merger()
    expected = 2.0 * 0.5 + 3.0 * 2.0
    assert torch.isclose(total_loss, torch.tensor(expected))

def test_forward_pass_arguments():
    """Test that arguments are propagated to all losses."""
    class ArgCheckLoss(nn.Module):
        def forward(self, x, y, flag=False):
            assert x.shape == y.shape, "Inputs must have the same shape"
            assert flag, "Flag must be True"
            return torch.tensor(1.0)

    losses = {'check_loss': ArgCheckLoss()}
    merger = MergingLossWrapper(losses)
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    # Should not raise errors
    merger(x, y, flag=True)

def test_empty_losses():
    """Test initialization with an empty losses dictionary."""
    merger = MergingLossWrapper({})
    assert len(merger.losses) == 0
    # Forward pass returns 0.0 (sum of no losses)
    assert merger() == torch.tensor(0.0)
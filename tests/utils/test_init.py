import pytest
import torch
import torch.nn as nn

from dl_toolkit.modules.utils.init_utils import init_net, init_weights


def test_init_requires_gain():
    """Test that 'normal' and 'uniform' require gain parameter."""
    net = nn.Conv2d(3, 3, 3)
    with pytest.raises(ValueError):
        init_weights(net, init_type="normal", gain=None)
    with pytest.raises(ValueError):
        init_weights(net, init_type="uniform", gain=None)


def test_batchnorm_init():
    """Test BatchNorm2d weight=1 and bias=0."""
    net = nn.BatchNorm2d(3)
    init_weights(net, init_type="kaiming_uniform")
    assert torch.allclose(net.weight.data, torch.ones(3))
    assert torch.allclose(net.bias.data, torch.zeros(3))


def test_invalid_init_type():
    """Test invalid initialization type raises error."""
    net = nn.Conv2d(3, 3, 3)
    with pytest.raises(NotImplementedError):
        init_weights(net, init_type="invalid_type")


def test_orthogonal_init():
    """Test orthogonal initialization produces orthogonal matrices."""
    net = nn.Linear(10, 10)
    init_weights(net, init_type="orthogonal", gain=1.0)
    W = net.weight.data
    product = torch.mm(W, W.T)
    assert torch.allclose(product, torch.eye(10), atol=1e-4)


def test_init_net():
    """Test init_net wrapper function."""
    net = nn.Conv2d(3, 3, 3)
    init_net(net, init_type="zeros")
    assert torch.all(net.weight.data == 0)

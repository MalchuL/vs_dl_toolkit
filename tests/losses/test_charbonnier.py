import torch

from dl_toolkit.modules.losses import CharbonnierLoss


def test_charbonnier_small():

    x = torch.ones(1000)
    y = torch.ones(1000) + torch.linspace(0, 0.001, 1000)
    loss = CharbonnierLoss(reduction="sum")
    y.requires_grad = True
    x.requires_grad = True

    loss_value = loss(x, y)
    print(loss_value)
    loss_value.backward()
    print(x.grad)
    print(y.grad)


def test_charbonnier_large():

    x = torch.ones(100)
    y = torch.ones(100) + torch.linspace(0, 1, 100)
    loss = CharbonnierLoss(reduction="sum")
    y.requires_grad = True
    x.requires_grad = True

    loss_value = loss(x, y)
    print(loss_value)
    loss_value.backward()
    print(x.grad)
    print(y.grad)

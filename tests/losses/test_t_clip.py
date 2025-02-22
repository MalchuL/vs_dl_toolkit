import torch

from dl_toolkit.modules.losses import TClipLoss


def test_t_clip():

    loss = TClipLoss(reduction="none")
    tensor = torch.tensor([-0.5, 0.5, 1.5])
    tensor.requires_grad = True
    loss_value = loss(tensor)
    assert torch.allclose(loss_value, torch.tensor([0.2500, 0.0000, 0.2500], dtype=torch.float32))
    loss_value.mean().backward()
    assert torch.allclose(
        tensor.grad.data,
        torch.tensor([-0.3333, 0.0000, 0.3333], dtype=tensor.grad.dtype),
        atol=1e-4,
    )

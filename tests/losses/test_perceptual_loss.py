import torch

from dl_toolkit.modules.losses.image.perceptual_loss import PerceptualLossSimple


def test_perceptual_loss():
    loss = PerceptualLossSimple(loss_type="l1")
    print(loss)
    x = torch.rand(1, 3, 256, 256)
    y = torch.rand(1, 3, 256, 256)
    out_loss_same = loss(x, x)
    assert torch.allclose(out_loss_same, torch.zeros_like(out_loss_same))
    out_loss_diff = loss(x, y)
    assert not torch.allclose(out_loss_diff, torch.zeros_like(out_loss_diff))

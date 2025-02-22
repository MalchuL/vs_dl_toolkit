import torch

from dl_toolkit.modules.losses.image.structure_loss import StructureLoss


def test_structure_loss():
    loss = StructureLoss()
    x = torch.randn(1, 3, 64, 64)
    y = torch.randn(1, 3, 64, 64)
    out_loss = loss(x, y)
    print(out_loss)

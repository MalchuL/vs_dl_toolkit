import pytest
import torch

from dl_toolkit.modules.losses.image.tv_loss import TVLoss


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_tv_loss(reduction):
    x = torch.randn(10, 3, 224, 224)
    loss = TVLoss(reduction=reduction)


def test_wrong_reduction():
    reduction = "wrong"
    with pytest.raises(ValueError):
        TVLoss(reduction=reduction)

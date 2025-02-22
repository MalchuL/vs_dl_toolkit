import logging

import torch

from dl_toolkit.modules.losses.classification.focal_loss import FocalLoss

logging.basicConfig(level=logging.WARNING)


def test_focal_loss():
    focal_loss = FocalLoss()
    tensor = torch.randn(2, 3, 256, 256)
    gt = (torch.randn(2, 3, 256, 256) > 0.0).float()
    loss = focal_loss(tensor, gt)
    print(loss)


def test_wrong_focal_loss():
    focal_loss = FocalLoss()
    for _ in range(10):
        tensor = torch.rand(2, 3, 256, 256)
        gt = (torch.randn(2, 3, 256, 256) > 0.0).float()
        loss = focal_loss(tensor, gt)
        print(loss)

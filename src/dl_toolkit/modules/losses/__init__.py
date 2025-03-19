from .classification.focal_loss import FocalLoss
from .distribution.kl_div_loss import KLDivergenceLoss
from .regression import CharbonnierLoss, TClipLoss
from .gan import GANLoss, SoftplusGANLoss, RPGANLoss
from .image import TVLoss, StructureLoss, PerceptualLoss, PerceptualLossSimple

from .loss_wrapper import LossWrapper
from .identity_loss import IdentityLoss

__all__ = ["FocalLoss", "KLDivergenceLoss", "CharbonnierLoss", "TClipLoss",
           "GANLoss", "SoftplusGANLoss", "RPGANLoss", "TVLoss", "StructureLoss", "PerceptualLoss",
           "PerceptualLossSimple", "LossWrapper", "IdentityLoss"]

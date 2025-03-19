import torch
from torch import nn

from dl_toolkit.modules.losses.utils.reduction import reduce_data
from dl_toolkit.modules.utils import logit


class RPGANLoss(nn.Module):
    """Relativistic Average GAN loss from "The GAN is dead; long live the GAN! A Modern Baseline GAN".

    This loss computes the relativistic average of discriminator outputs for real and fake samples,
    applying a specified criterion (default: `nn.Softplus`). Supports logit conversion and loss reduction.

    Args:
        rel_avg_gan (bool, optional): If `False`, applies RpGAN loss from paper.
            If `True`, applies relativistic average GAN loss. Defaults to `False`.
        is_logit (bool, optional): If `True`, assumes inputs are logits and applies no conversion.
            If `False`, converts inputs to logits using `logit()`. Defaults to `True`.
        reduction (str, optional): Reduction method for aggregating loss values. Options: "mean", "sum", "none".
            Defaults to "mean".

    Reference:
        Jolicoeur-Martineau, A. (2019). The relativistic discriminator: A key element missing from standard GAN.
        arXiv preprint arXiv:1807.00734.
    """

    def __init__(self, rel_avg_gan: bool = False, is_logit: bool = True,
                 reduction: str = "mean"):
        """Initializes the RPGANLoss with criterion, logit handling, and reduction method."""
        super().__init__()
        self.criterion = nn.Softplus()
        self.rel_avg_gan = rel_avg_gan
        self.is_logit = is_logit
        self.reduction = reduction

    def forward(self, fake_pred: torch.Tensor, real_pred: torch.Tensor,
                is_generator_loss: bool) -> torch.Tensor:
        """Computes the relativistic GAN loss.
           Please interpret fake as generated samples and real as real samples. Do not switch
           the arguments.
        Args:
            fake_pred (torch.Tensor): Discriminator predictions for fake samples.
            real_pred (torch.Tensor): Discriminator predictions for real samples.
            is_generator_loss (bool): If `True`, computes loss for generator training.
                If `False`, computes loss for discriminator training.

        Returns:
            torch.Tensor: Computed loss value after reduction.
        """
        if not self.is_logit:
            fake_pred = logit(fake_pred)
            real_pred = logit(real_pred)

        if self.rel_avg_gan:
            # Relativistic Average GAN loss formula:
            assert len(real_pred.shape) >= 2
            assert len(fake_pred.shape) >= 2

            if is_generator_loss:
                relativistic_logits = fake_pred - real_pred.mean(0, keepdim=True)
                loss = self.criterion(-relativistic_logits)
            else:
                real_rel_logits = real_pred - fake_pred.mean(0, keepdim=True)
                fake_rel_logits = fake_pred - real_pred.mean(0, keepdim=True)
                loss = (self.criterion(-real_rel_logits) + self.criterion(fake_rel_logits)) / 2
        else:
            # Relativistic Average GAN loss from "The GAN is dead; long live the GAN! A Modern
            # Baseline GAN, doesn't reduce logits"
            if is_generator_loss:
                relativistic_logits = fake_pred - real_pred
            else:
                relativistic_logits = real_pred - fake_pred
            loss = self.criterion(-relativistic_logits)
        return reduce_data(loss, reduction=self.reduction)

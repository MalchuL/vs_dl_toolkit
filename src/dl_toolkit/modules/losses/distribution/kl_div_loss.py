from typing import Tuple, Union

import torch

from dl_toolkit.modules.utils.distribution import DiagonalGaussianDistribution
from dl_toolkit.modules.toolkit_module import ToolkitModule


class KLDivergenceLoss(ToolkitModule):
    VERSION = "1.0.0"

    def __init__(self, dims=(1, 2, 3)):
        """
        Loss for KL divergence
        :param dims: tuple of ints, representing the dimensions of
                     single sample (excluded batch dimension)
        """
        super().__init__()
        self.dims = dims

    def forward(self, x: Union[Tuple, torch.Tensor]):
        distribution = DiagonalGaussianDistribution(x, deterministic=False, generator=None)
        return distribution.kl(dim=self.dims)

import math

import numpy as np
import torch

from dl_toolkit.modules.toolkit_module import ToolkitModule
from dl_toolkit.utils.logging import logger

SQRT_2 = math.sqrt(2)


class FourierEmbedding(ToolkitModule):
    """Implements Fourier feature embedding using random frequency projections.

    Based on paper: Fourier Features Let Networks Learn
                    High Frequency Functions in Low Dimensional Domains
    This layer transforms input features into a higher dimensional space using
    random Fourier features, following the architecture commonly used in
    neural tangent kernel (NTK) aware initialization schemes.

    Args:
        num_channels (torch.Tensor): Frequency parameters of shape (num_channels//2,)
        scale (torch.Tensor): Fixed scaling factor of 2π
    """

    def __init__(self, num_channels, scale=16):
        """Initializes Fourier embedding layer with random frequencies.

        Args:
            num_channels (int): Number of output channels. Should be even for
                symmetric frequency distribution.
            scale (float, optional): Scaling factor for frequency initialization.
                Defaults to 16. Higher values emphasize higher frequencies.
        """
        super().__init__()
        freqs = torch.randn(num_channels // 2) * scale
        multiplier = torch.tensor(2 * np.pi).to(freqs.dtype)

        self.register_buffer("freqs", freqs)
        self.register_buffer("multiplier", multiplier)

    def forward(self, timesteps):
        """Computes Fourier embedding for input features.

        Args:
            timesteps (torch.Tensor): Input tensor of shape (B) that represents timestep (in float between 0 and 1)

        Returns:
            torch.Tensor: Output tensor of shape (B, num_channels) containing
                concatenated cosine and sine features scaled by √2
        """
        assert timesteps.dim() == 1, "Only supports 1D input tensors"
        if timesteps.max() > 1:
            logger.warning(
                f"Input tensor {timesteps.max()} has values greater than 1. "
                "For Fouier embedding, input values should be between 0 and 1."
            )
        x = timesteps.outer(self.freqs * self.multiplier)
        x = torch.cat([x.cos(), x.sin()], dim=1) * SQRT_2
        return x

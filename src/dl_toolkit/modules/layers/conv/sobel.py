import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dl_toolkit.modules.toolkit_module import ToolkitModule


class SobelFilter(ToolkitModule):
    """A Sobel filter module for edge detection and gradient computation.

    This module applies Sobel filters to detect edges in both horizontal and vertical directions,
    with options to return gradient magnitudes or individual components.
    """

    def __init__(
        self,
        k_sobel=3,
        only_edges=False,
        *,
        use_padding=True,
        reduction_weight=(0.299, 0.587, 0.114),
        eps=1e-4,
    ):
        """Initialize Sobel filter with configurable parameters.

        Args:
            k_sobel (int, optional): Size of the Sobel kernel. Must be odd. Defaults to 3.
            only_edges (bool, optional): If True, returns gradient magnitude. If False, returns
                both x and y gradients. Defaults to False.
            use_padding (bool, optional): Maintains input spatial dimensions using reflection
                padding. Defaults to True.
            reduction_weight (tuple, optional): RGB to grayscale conversion weights.
                Defaults to (0.299, 0.587, 0.114).
            eps (float, optional): Small value to prevent sqrt(0) in gradient magnitude.
                Defaults to 1e-4.
        """
        super().__init__()
        sobel_2D = self.get_sobel_kernel(k_sobel)
        self.register_buffer(
            "sobel_filter_x", torch.tensor(sobel_2D.tolist()).view(1, 1, k_sobel, k_sobel)
        )
        self.register_buffer(
            "sobel_filter_y", torch.tensor(sobel_2D.T.tolist()).view(1, 1, k_sobel, k_sobel)
        )

        self.only_edges = only_edges
        self.eps_squared = eps**2

        self.padding = nn.ReflectionPad2d(k_sobel // 2) if use_padding else nn.Identity()

        self.register_buffer(
            "reduction_weight", torch.tensor(reduction_weight).view(1, len(reduction_weight), 1, 1)
        )

    def apply(self, fn):
        """Override apply method to prevent parameter re-initialization."""
        return

    def rgb2gray(self, tensor):
        """Convert RGB tensor to grayscale using predefined weights.

        Args:
            tensor (torch.Tensor): Input tensor of shape (B, 3, H, W)

        Returns:
            torch.Tensor: Grayscale tensor of shape (B, 1, H, W)
        """
        return torch.sum(tensor * self.reduction_weight, 1, keepdim=True)

    @staticmethod
    def get_sobel_kernel(k=3):
        """Generate Sobel kernel matrix.

        Args:
            k (int, optional): Kernel size. Must be odd. Defaults to 3.

        Returns:
            np.ndarray: Sobel kernel matrix of shape (k, k)
        """
        range_coords = np.linspace(-(k // 2), k // 2, k)
        # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range_coords, range_coords)
        sobel_2D_numerator = x
        sobel_2D_denominator = x**2 + y**2
        sobel_2D_denominator[:, k // 2] = 1  # Avoid division by zero
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator
        return sobel_2D

    @property
    def channels(self):
        """Get number of output channels.

        Returns:
            int: 1 if only edge magnitude, 2 if both gradients
        """
        return 1 if self.only_edges else 2

    def forward(self, x):
        """Process input through Sobel filtering pipeline.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Edge features. Shape (B, 1, H, W) if only_edges=True,
                else (B, 2, H, W)
        """
        x = self.rgb2gray(x)
        x = self.padding(x)
        # We took the absolute value of the gradient to avoid negative values
        grad_x = torch.abs(F.conv2d(x, self.sobel_filter_x))
        grad_y = torch.abs(F.conv2d(x, self.sobel_filter_y))

        if self.only_edges:
            return torch.sqrt(grad_x**2 + grad_y**2 + self.eps_squared)
        return torch.cat([grad_x, grad_y], dim=1)

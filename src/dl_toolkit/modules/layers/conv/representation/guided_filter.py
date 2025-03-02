import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# Fdgf from https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Learning_to_Cartoonize_Using_White-Box_Cartoon_Representations_CVPR_2020_paper.pdf
class GuidedFilter(nn.Module):
    """Guided filter implementation from White-Box Cartoonization.

    Reference: https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Learning_to_Cartoonize_Using_White-Box_Cartoon_Representations_CVPR_2020_paper.pdf

    Args:
        r (int): Filter radius (kernel size = 2r+1)
        input_channels (int): Number of input channels
        eps (float): Regularization parameter for numerical stability

    Attributes:
        r (int): Current filter radius
        input_channels (int): Configured input channels
        eps (float): Current epsilon value
        box_kernel (torch.Tensor): Precomputed box filter kernel
        N_box_kernel (torch.Tensor): Normalization kernel
    """

    def __init__(self, r=1, input_channels=3, eps=1e-2):
        """Initialize guided filter with specified parameters."""
        super().__init__()
        self.r = r
        self.input_channels = input_channels
        self.eps = eps
        self.register_buffer("box_kernel", self.calculate_box_filter(self.r, self.input_channels))
        self.register_buffer("N_box_kernel", self.calculate_box_filter(self.r, 1))

    @staticmethod
    def calculate_box_filter(r: int, ch: int) -> torch.Tensor:
        """Generate box filter kernel.

        Args:
            r (int): Filter radius
            ch (int): Number of channels

        Returns:
            torch.Tensor: Box filter kernel tensor of shape (ch, 1, 2r+1, 2r+1)
        """
        weight = 1 / ((2 * r + 1) ** 2)
        return weight * torch.ones([ch, 1, 2 * r + 1, 2 * r + 1], dtype=torch.float32)

    def box_filter(self, x: torch.Tensor, channels: int = None) -> torch.Tensor:
        """Apply box filter to input tensor.

        Args:
            x (torch.Tensor): Input tensor (NCHW format)
            channels (int, optional): Number of channels for kernel selection

        Returns:
            torch.Tensor: Filtered output tensor
        """
        return F.conv2d(
            x,
            self.N_box_kernel if channels == 1 else self.box_kernel,
            bias=None,
            stride=1,
            padding="same",
            groups=self.input_channels if channels is None else channels,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply guided filtering operation.

        Args:
            x (torch.Tensor): Guidance image (NCHW format)
            y (torch.Tensor): Input image to filter (NCHW format)

        Returns:
            torch.Tensor: Filtered output image

        Raises:
            RuntimeError: If input tensors have mismatched dimensions
        """
        N, C, H, W = x.shape
        if y.shape != x.shape:
            raise RuntimeError("Input and guidance images must have same shape")

        N = self.box_filter(torch.ones(1, 1, H, W, dtype=x.dtype, device=x.device), channels=1)
        mean_x = self.box_filter(x) / N
        mean_y = self.box_filter(y) / N

        cov_xy = self.box_filter(x * y) / N - mean_x * mean_y
        var_x = self.box_filter(x * x) / N - mean_x * mean_x

        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x

        mean_A = self.box_filter(A) / N
        mean_b = self.box_filter(b) / N

        return mean_A * x + mean_b

    def get_num_channels(self) -> int:
        """Get number of output channels.

        Returns:
            int: Always returns 3 for RGB output
        """
        return 3

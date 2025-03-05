import numpy as np
import torch
from torch.nn import functional as F

from dl_toolkit.modules.toolkit_module import ToolkitModule


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
    else:
        raise ValueError("Wrong kernel size")

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


def gkern(kernel: int = 5, sig: float = 1.0) -> np.ndarray:
    """Generate 1D Gaussian kernel array.

    Args:
        kernel (int): Kernel length (should be odd)
        sig (float): Standard deviation of Gaussian distribution

    Returns:
        np.ndarray: Normalized 1D Gaussian kernel array
    """
    ax = np.linspace(-(kernel - 1) / 2.0, (kernel - 1) / 2.0, kernel)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    return gauss / gauss.sum()


class USMSharp(ToolkitModule):
    """Unsharp Mask sharpening module with adaptive thresholding.

    Reference: https://arxiv.org/pdf/2107.10833

    Args:
        radius (int): Gaussian blur radius (kernel size = 2*radius + 1)
        sigma (float): Gaussian kernel standard deviation

    Attributes:
        radius (int): Current blur radius
        kernel (Tensor): Precomputed 2D Gaussian kernel
    """

    def __init__(self, radius: int = 50, sigma: float = 1):
        super().__init__()
        if radius % 2 == 0:
            radius += 1
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.radius = radius
        kernel = gkern(radius, sigma)

        kernel = np.outer(kernel, kernel)  # Create 2D kernel
        self.register_buffer("kernel", torch.FloatTensor(kernel).unsqueeze(0))

    def forward(
        self, img: torch.Tensor, weight: float = 0.5, threshold: float = 10
    ) -> torch.Tensor:
        """Apply adaptive unsharp mask sharpening.

        Args:
            img (Tensor): Input image tensor in NCHW format (0-1 range)
            weight (float): Sharpening strength (0-1)
            threshold (float): Residual threshold for mask creation

        Returns:
            Tensor: Sharpened image in same format as input
        """
        blur = filter2D(img, self.kernel)
        residual = img - blur

        # Create adaptive mask
        mask = (torch.abs(residual) * 255 > threshold).type_as(img)
        soft_mask = filter2D(mask, self.kernel)

        # Apply sharpening with adaptive blending
        sharp = img + weight * residual
        return torch.clamp(soft_mask * sharp + (1 - soft_mask) * img, 0, 1)

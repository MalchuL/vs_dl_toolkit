import torch
from torch import nn

from .constants import BIAS, PADDING_MODE


class SeparableConv2d(nn.Module):
    """Implements a separable convolution layer with different architecture modes.

    Separable convolutions factorize standard convolutions into depthwise and pointwise
    operations for improved efficiency. Supports two different architectural variants:
    - 'default': Depthwise convolution followed by pointwise convolution
    - 'inception': Pointwise convolution followed by depthwise convolution

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding size. Defaults to kernel_size//2 if None.
        dilation (int, optional): Dilation rate. Defaults to 1.
        groups (int, optional): Number of blocked connections from input to output. Defaults to 1.
        bias (bool, optional): Whether to use bias terms. Defaults to BIAS constant.
        padding_mode (str, optional): Padding mode. Defaults to PADDING_MODE constant.
        sep_mode (str, optional): Architecture variant ('default' or 'inception'). Defaults to 'default'.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = BIAS,
                 padding_mode: str = PADDING_MODE,
                 sep_mode='default'):
        super().__init__()
        if sep_mode not in ['default', 'inception']:
            raise ValueError(f'Invalid separable convolution mode: {sep_mode}')
        self.sep_mode = sep_mode

        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if padding is not None else self.kernel_size // 2
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

        self.module = self.construct_module()

    def construct_module(self) -> nn.Sequential:
        """Builds the sequential convolution modules based on configuration.

        Returns:
            nn.Sequential: Sequential container with configured convolution layers
        """
        if self.kernel_size == 1:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=self.stride,
                    groups=self.groups,
                    bias=self.bias
                )
            )

        if self.sep_mode == 'default':
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.in_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.in_channels,
                    bias=False,
                    padding_mode=self.padding_mode
                ),
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    groups=self.groups,
                    bias=self.bias
                )
            )
        elif self.sep_mode == 'inception':
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    groups=self.groups,
                    bias=False
                ),
                nn.Conv2d(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.out_channels,
                    bias=self.bias,
                    padding_mode=self.padding_mode
                )
            )
        else:
            raise ValueError(f'Invalid separable convolution mode: {self.sep_mode}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the separable convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H', W')
        """
        return self.module(x)

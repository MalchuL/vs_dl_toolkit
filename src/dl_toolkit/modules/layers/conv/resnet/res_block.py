from torch import nn

from ..convbnrelu import Conv2dBNReLU
from ..separable_conv2d import SeparableConv2d
from ...activation import get_act


class ResidualBlock(nn.Module):
    """Residual block with optional depthwise separable convolutions and post-residual activation.

    This block consists of two Conv2dBNReLU layers followed by a residual connection. If the input
    and output dimensions differ, a 1x1 convolution is used to match dimensions. Supports zero-initializing
    the final convolution layer and adding an activation after the residual addition.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding (int, optional): Padding added to all sides of the input. If None, set to
            `(kernel_size - 1) // 2`. Defaults to None.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        dilation (int, optional): Spacing between kernel elements. Only dilation=1 is supported.
            Defaults to 1.
        groups (int, optional): Number of blocked connections from input to output channels.
            Defaults to 1.
        separable (bool, optional): If True, uses depthwise separable convolutions. Defaults to False.
        zero_init_residual (bool, optional): If True, zero-initializes the last convolution layer.
            Defaults to False.
        last_act (bool, optional): If True, applies activation after the residual addition.
            Defaults to False.
        norm_layer (str, optional): Normalization layer type. Can be 'batch', 'none', or None.
            Defaults to 'batch'.
        act_layer (str, optional): Activation layer type. Defaults to 'relu'.

    Raises:
        NotImplementedError: If `dilation > 1` is provided.
    """

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=None,
                 stride=1,
                 dilation=1,
                 groups=1,
                 separable=False,
                 zero_init_residual=False,
                 last_act=False,
                 norm_layer='batch',
                 act_layer='relu'):
        super(ResidualBlock, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = (kernel_size - 1) // 2
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 is not supported!')
        use_norm_layer = norm_layer not in ['none', None]
        self.conv1 = Conv2dBNReLU(in_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  bias=not use_norm_layer,
                                  dilation=dilation,
                                  groups=groups,
                                  separable=separable,
                                  norm_layer=norm_layer,
                                  act_layer=act_layer)
        self.conv2 = Conv2dBNReLU(out_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  bias=not use_norm_layer,
                                  dilation=dilation,
                                  groups=groups,
                                  separable=separable,
                                  norm_layer=norm_layer,
                                  act_layer=None)

        self.downsample = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=1,
                                        stride=stride,
                                        bias=False)

        if zero_init_residual:
            if use_norm_layer:
                nn.init.zeros_(self.conv2.norm.weight.data)
                nn.init.zeros_(self.conv2.norm.bias.data)
            else:
                conv = self.conv2.conv
                if isinstance(conv, SeparableConv2d):
                    conv = conv.module[-1]
                nn.init.zeros_(conv.weight.data)
                nn.init.zeros_(conv.bias.data)

        self.out_channels = out_channels
        self.last_act = nn.Identity()
        if last_act:
            self.last_act = get_act(act_layer)(inplace=False, in_channels=out_channels)

    def forward(self, x):
        """Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H', W') with optional activation.
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        residual = self.downsample(residual)
        out = residual + out
        out = self.last_act(out)
        return out

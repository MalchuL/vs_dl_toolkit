from torch import nn

from dl_toolkit.modules.toolkit_module import ToolkitModule

from ..convbnrelu import BNReLUConv2d
from ..separable_conv2d import SeparableConv2d


class ResidualBlockV2(ToolkitModule):
    """Residual block with pre-activation structure (ResNet V2).

    This block consists of two BNReLUConv2d layers followed by a residual connection.
    If the number of input channels differs from output channels or stride != 1,
    a 1x1 convolution is used to match dimensions for the residual connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding (int, optional): Padding added to all four sides of the input. If None,
            padding is set to (kernel_size - 1) // 2. Defaults to None.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        dilation (int, optional): Spacing between kernel elements. Only dilation=1 is supported.
            Defaults to 1.
        groups (int, optional): Number of blocked connections from input to output channels.
            Defaults to 1.
        separable (bool, optional): If True, uses depthwise separable convolutions. Defaults to False.
        zero_init_residual (bool, optional): If True, initializes the weights of the last
            convolution layer to zero. Defaults to False.
        norm_layer (str, optional): Type of normalization layer. Can be 'batch', 'none', or None.
            Defaults to 'batch'.
        act_layer (str, optional): Type of activation layer. Defaults to 'relu'.

    Raises:
        NotImplementedError: If dilation > 1 is provided.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=None,
        stride=1,
        dilation=1,
        groups=1,
        separable=False,
        zero_init_residual=False,
        norm_layer="batch",
        act_layer="relu",
    ):
        """Initializes ResidualBlockV2 with specified parameters."""
        super().__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = (kernel_size - 1) // 2
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 is not supported!")
        use_norm_layer = norm_layer not in ["none", None]
        self.conv1 = BNReLUConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_norm_layer,
            dilation=dilation,
            groups=groups,
            separable=separable,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.conv2 = BNReLUConv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=not use_norm_layer,
            dilation=dilation,
            groups=groups,
            separable=separable,
            norm_layer=norm_layer,
            act_layer=None,
        )

        self.downsample = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
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

    def forward(self, x):
        """Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H', W').
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        residual = self.downsample(residual)
        out = residual + out
        return out

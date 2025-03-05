from torch import nn

from .constants import BIAS, PADDING_MODE
from .separable_conv2d import SeparableConv2d
from ..activation import get_act
from ..norm_layers import get_norm_layer


class Conv2dBNReLU(nn.Module):
    """
    A sequential block comprising a 2D convolution (regular or separable), batch normalization,
    and an activation layer, applied in the order: Conv2d -> Norm -> Activation.

    Attributes:
        module (nn.Sequential): Sequential container holding the convolution, normalization, and activation layers.

    Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolution kernel. Defaults to 3.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding added to all sides of the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input to output channels. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to BIAS.
        padding_mode (str, optional): Type of padding. Defaults to PADDING_MODE.
        separable (bool, optional): If True, uses SeparableConv2d instead of Conv2d. Defaults to False.
        norm_layer (str, optional): Type of normalization layer (e.g., 'batch', 'instance'). Defaults to 'batch'.
        act_layer (str, optional): Type of activation function (e.g., 'relu', 'leaky_relu'). Defaults to 'relu'.
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
                 separable: bool = False,
                 # Parameters for norm layers
                 norm_layer: str = 'batch',
                 act_layer: str = 'relu',
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.padding_mode = padding_mode
        self.dilation = dilation
        self.groups = groups

        self.separable = separable

        self.norm_layer = norm_layer
        self.act_layer = act_layer

        self.conv = self.get_conv_layer()
        self.norm = get_norm_layer(self.norm_layer)(self.out_channels)
        self.act = get_act(self.act_layer)(inplace=True, in_channels=self.out_channels)

    def get_conv_layer(self):
        """
        Creates the convolution layer based on the configuration.

        Returns:
            nn.Module: SeparableConv2d if `separable` is True, otherwise Conv2d.
        """
        if self.separable:
            conv_layer = SeparableConv2d(in_channels=self.in_channels,
                                         out_channels=self.out_channels,
                                         kernel_size=self.kernel_size, stride=self.stride,
                                         padding=self.padding,
                                         dilation=self.dilation, groups=self.groups,
                                         bias=self.bias,
                                         padding_mode=self.padding_mode)
        else:
            conv_layer = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                   kernel_size=self.kernel_size, stride=self.stride,
                                   padding=self.padding,
                                   dilation=self.dilation, groups=self.groups, bias=self.bias,
                                   padding_mode=self.padding_mode)
        return conv_layer

    def forward(self, x):
        """
        Defines the forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the sequential layers.
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class BNReLUConv2d(nn.Module):
    """
    A sequential block comprising batch normalization, activation, and a 2D convolution (regular or separable),
    applied in the order: Norm -> Activation -> Conv2d.

    Attributes:
        module (nn.Sequential): Sequential container holding the normalization, activation, and convolution layers.

    Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolution kernel. Defaults to 3.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding added to all sides of the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input to output channels. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to BIAS.
        padding_mode (str, optional): Type of padding. Defaults to PADDING_MODE.
        separable (bool, optional): If True, uses SeparableConv2d instead of Conv2d. Defaults to False.
        norm_layer (str, optional): Type of normalization layer (e.g., 'batch', 'instance'). Defaults to 'batch'.
        act_layer (str, optional): Type of activation function (e.g., 'relu', 'leaky_relu'). Defaults to 'relu'.
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
                 separable: bool = False,
                 # Parameters for norm layers
                 norm_layer: str | None = 'batch',
                 act_layer: str | None = 'relu',
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.padding_mode = padding_mode
        self.dilation = dilation
        self.groups = groups

        self.separable = separable

        self.norm_layer = norm_layer
        self.act_layer = act_layer

        self.norm = get_norm_layer(self.norm_layer)(self.in_channels)
        self.act = get_act(self.act_layer)(inplace=True, in_channels=self.in_channels)
        self.conv = self.get_conv_layer()

    def get_conv_layer(self):
        """
        Creates the convolution layer based on the configuration.

        Returns:
            nn.Module: SeparableConv2d if `separable` is True, otherwise Conv2d.
        """
        if self.separable:
            conv_layer = SeparableConv2d(in_channels=self.in_channels,
                                         out_channels=self.out_channels,
                                         kernel_size=self.kernel_size, stride=self.stride,
                                         padding=self.padding,
                                         dilation=self.dilation, groups=self.groups,
                                         bias=self.bias,
                                         padding_mode=self.padding_mode)
        else:
            conv_layer = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                   kernel_size=self.kernel_size, stride=self.stride,
                                   padding=self.padding,
                                   dilation=self.dilation, groups=self.groups, bias=self.bias,
                                   padding_mode=self.padding_mode)
        return conv_layer

    def forward(self, x):
        """
        Defines the forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the sequential layers.
        """
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x

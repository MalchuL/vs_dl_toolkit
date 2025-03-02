import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dl_toolkit.modules.toolkit_module import ToolkitModule


class BlurPool(ToolkitModule):
    """Anti-aliased downsampling from https://arxiv.org/abs/1904.11486"""

    def __init__(self, channels, filt_size=3, stride=2):
        super().__init__()
        self.stride = stride
        self.pad = [filt_size // 2] * 4  # Padding for both sides

        # Create fixed blur kernel
        if filt_size == 3:
            a = torch.tensor([1.0, 2.0, 1.0])
        elif filt_size == 5:
            a = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        else:
            raise ValueError("Only filter sizes 3 and 5 are supported")
        a = a[:, None] * a[None, :]
        a = a / a.square().sum().sqrt()
        self.register_buffer("kernel", a[None, None, :, :].repeat(channels, 1, 1, 1))

    def forward(self, x):
        x = F.pad(x, self.pad, mode="reflect")
        return F.conv2d(x, self.kernel, stride=self.stride, groups=x.shape[1])


class AvgPoolNorm(ToolkitModule):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(2, stride=2)
        self.scale = 2  # sqrt(4) because of variance scaling

    def forward(self, x):
        x = self.pool(x)
        return x * self.scale


class Downsample(ToolkitModule):
    """Improved downsampling block with anti-aliasing and proper initialization

    Features:
    - Anti-aliased blur pooling for better shift invariance
    - Proper weight initialization accounting for pooling scale
    - Optional channel expansion
    - Correct variance scaling through network
    """

    def __init__(self, in_channels, out_channels=None, pool_type="blur", depthwise=False):
        super().__init__()
        out_channels = out_channels or in_channels

        # Anti-aliased downsampling
        if pool_type == "blur":
            self.pool = BlurPool(in_channels, filt_size=3)
        elif pool_type == "avg":
            self.pool = AvgPoolNorm()
        else:
            raise ValueError(f"Unknown downsampling type={pool_type}")

        # Channel transformation with variance-preserving initialization
        if depthwise:
            if in_channels != out_channels:
                raise ValueError(
                    "Depthwise convolution must have the same number of "
                    "input and output channels"
                )
            groups = in_channels
        else:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=groups, bias=False)

        # Initialize weights accounting for 0.5 scale from blur pooling
        nn.init.kaiming_uniform_(self.conv.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)

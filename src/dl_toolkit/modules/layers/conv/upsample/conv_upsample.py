import torch.nn.functional as F
from torch import nn

from dl_toolkit.modules.toolkit_module import ToolkitModule


class Upsample(ToolkitModule):
    def __init__(self, in_channels, out_channels=None, depthwise=False,
                 interpolate_mode="nearest"):
        super().__init__()
        self.interpolate_mode = interpolate_mode
        out_channels = out_channels or in_channels
        if depthwise:
            groups = in_channels
        else:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=groups, bias=False)

    def forward(self, x):
        # Firstly upsample, after activate
        x = F.interpolate(x, scale_factor=2.0, mode=self.interpolate_mode)
        x = self.conv(x)
        return x

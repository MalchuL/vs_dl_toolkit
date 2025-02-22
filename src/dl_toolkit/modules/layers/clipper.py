from typing import Iterable

from dl_toolkit.modules.toolkit_module import ToolkitModule
from dl_toolkit.modules.utils.clipping_utils import z_clip

_CLIP_SCORE = 3


class Clipper(ToolkitModule):
    VERSION = "1.0.0"
    CHANNELS: Iterable[int] | None = None

    def __init__(self, z_score: float = _CLIP_SCORE):
        super().__init__()
        self.z_score = z_score

    def forward(self, x):
        return z_clip(x, z_value=self.z_score, dims=self.CHANNELS)


class ClipperChannelwise1D(Clipper):
    CHANNELS = (2,)


class ClipperChannelwise2D(Clipper):
    CHANNELS = (2, 3)


class ClipperWrapper(ToolkitModule):
    def __init__(self, module: ToolkitModule, clipper: Clipper):
        """
        Wrapper for clipping module outputs.
        Args:
            module (nn.Module): Module to be wrapped and output clipped.
            clipper (clipper): Clipping module, which takes the output of the module and clips it.
                You can use same clipper in several wrappers.
        """
        super().__init__()
        self.module = module
        self.clipper = clipper

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return self.clipper(x)

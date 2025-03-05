from dl_toolkit.modules.layers.mult_alpha import MultAlpha
from dl_toolkit.modules.toolkit_module import ToolkitModule


class Residual(ToolkitModule):
    """
    Implements  x + f(x) * alpha
    """

    def __init__(self, module, init_blend: float = 0.0, use_alpha: bool = True):
        super().__init__()
        self.use_alpha = use_alpha
        if self.use_alpha:
            self.module = MultAlpha(module, init_blend=init_blend)
        else:
            # Don't use alpha, just return the module, to save memory and time.
            self.module = module

    def forward(self, x):
        return x + self.module(x)

    def extra_repr(self):
        return f"module={self.module}"

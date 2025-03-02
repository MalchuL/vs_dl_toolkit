from dl_toolkit.modules.layers.mult_alpha import MultAlpha
from dl_toolkit.modules.toolkit_module import ToolkitModule


class Residual(ToolkitModule):
    """
    Implements  x + f(x) * alpha
    """

    def __init__(self, module, init_blend=0.0):
        super().__init__()
        self.mult_alpha = MultAlpha(module, init_blend=init_blend)

    def forward(self, x):
        return x + self.mult_alpha(x)

    def extra_repr(self):
        return f"module={self.mult_alpha}"

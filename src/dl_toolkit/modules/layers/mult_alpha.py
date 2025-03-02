import torch
import torch.nn as nn

from dl_toolkit.modules.toolkit_module import ToolkitModule


class MultAlpha(ToolkitModule):
    """
    Implements f(x) * alpha, usefull in cases y + f(x) * alpha where you don't want to keep parameters in the model
    """

    def __init__(self, module, init_blend=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_blend)))
        self.module = module

    def forward(self, *args, **kwargs):
        alpha = self.alpha
        return self.module(*args, **kwargs) * alpha

    def extra_repr(self):
        return 'blend_value={}'.format(self.alpha.item())

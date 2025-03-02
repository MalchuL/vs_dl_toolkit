import torch

from dl_toolkit.modules.toolkit_module import ToolkitModule


# TODO add pydoc for this class
class CoordConv(ToolkitModule):
    def __init__(self, with_r=False):
        super(CoordConv, self).__init__()
        self.with_r = with_r
        # For buffering tensors
        self._last_shape = None
        self._buffered_tensor = None

    @property
    def channels(self):
        return 2 + int(self.with_r)

    def create_coords(self, input_tensor):
        batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
        xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

        xx_range = torch.arange(dim_y, dtype=torch.int32)
        yy_range = torch.arange(dim_x, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        # transpose y
        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (dim_y - 1)
        yy_channel = yy_channel.float() / (dim_x - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

        xx_channel = xx_channel.type_as(input_tensor)
        yy_channel = yy_channel.type_as(input_tensor)
        out = torch.cat([xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            out = torch.cat([out, rr], dim=1)
        return out

    def forward(self, x):
        if x.shape == self._last_shape:
            out = self._buffered_tensor
        else:
            out = self.create_coords(x)
            self._last_shape = x.shape
            self._buffered_tensor = out
        return out

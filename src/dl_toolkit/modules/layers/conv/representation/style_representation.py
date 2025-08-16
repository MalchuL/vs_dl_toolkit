import torch
from torch import nn

from dl_toolkit.modules.layers.conv.representation.color_shift import ColorShift
from dl_toolkit.modules.layers.conv.coord_conv import CoordConv
from dl_toolkit.modules.layers.conv.representation.guided_filter import GuidedFilter
from dl_toolkit.modules.layers.conv.sobel import SobelFilter


# Style representation module https://arxiv.org/pdf/2207.02426.pdf
class StyleRepresentation(nn.Module):
    """A module for extracting and combining different style representations of an image.

    This module implements the style representation approach from the paper:
    "Designing a Better Style Transfer Network by Emphasizing the Style Representation"
    (https://arxiv.org/pdf/2207.02426.pdf)

    The module can combine multiple style representation layers including:
    - color_shift: Adjusts color distribution
    - sobel: Extracts edge information
    - surface: Applies guided filtering for surface details
    - identity: Preserves original input

    Args:
        layers (tuple, optional): Sequence of layer names to use. Valid names are:
            'color_shift', 'sobel', 'surface', 'identity'. Defaults to ().
        weight_mode (str, optional): Weight mode for ColorShift layer. Defaults to 'uniform'.
        add_coords (bool, optional): Whether to append coordinate channels. Defaults to False.
        r (int, optional): Radius parameter for GuidedFilter. Defaults to 5.
        eps (float, optional): Epsilon parameter for GuidedFilter. Defaults to 2e-1.

    Example:
        >>> style_rep = StyleRepresentation(layers=('color_shift', 'sobel', 'identity'))
        >>> x = torch.randn(1, 3, 64, 64)
        >>> out = style_rep(x)  # Shape: [1, 9, 64, 64] (3 layers * 3 channels)
    """
    def __init__(self, layers=(), *args, weight_mode='uniform', add_coords=False, r=5, eps=2e-1):
        super().__init__()
        if len(layers) == 0:
            raise ValueError("At least one layer must be specified")
        self.layer_names = layers
        self.coord_conv = CoordConv()
        self.add_coords = add_coords
        self.layers = nn.ModuleDict({'color_shift': ColorShift(weight_mode=weight_mode),
                                     'sobel': SobelFilter(only_edges=True),
                                     'surface': GuidedFilter(r=r, eps=eps),
                                     'identity': nn.Identity()})
        for layer_name in layers:
            if layer_name not in self.layers:
                raise ValueError(f"Invalid layer name: {layer_name}")

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where C is 
            the number of channels and must be 3.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
        """
        if x.shape[1] != 3:
            raise ValueError("Input tensor must have 3 channels")
        layers = []
        for layer_name in self.layer_names:
            if layer_name == 'surface':
                out_layer = self.layers[layer_name](x, x)
            else:
                out_layer = self.layers[layer_name](x)
            if out_layer.shape[1] == 1:
                out_layer = out_layer.repeat(1, 3, 1, 1)
            if self.add_coords:
                out_layer = torch.cat([out_layer, self.coord_conv(out_layer)], dim=1)
            layers.append(out_layer)
        out = torch.cat(layers, dim=1)
        return out

    @property
    def channels(self):  # Estimated number of channels
        additional_channels = 0
        if self.add_coords:
            additional_channels += 2
        return len(self.layer_names) * (3 + additional_channels)

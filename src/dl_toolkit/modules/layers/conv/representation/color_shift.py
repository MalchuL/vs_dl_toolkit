import torch

from dl_toolkit.modules.toolkit_module import ToolkitModule


class ColorShift(ToolkitModule):
    """Color shift transformation module from White-Box Cartoonization.

    Implements channel-wise weighting from:
    "Learning to Cartoonize Using White-Box Cartoon Representations" (CVPR 2020)

    Args:
        weight_mode (str): Weight sampling mode - 'normal' or 'uniform'
        is_repeat (bool): Whether to repeat single channel output to RGB

    Attributes:
        is_repeat (bool): Repeat output channel flag
        weight_mode (str): Current weight sampling mode
    """

    def __init__(self, weight_mode="uniform", is_repeat=True):
        super().__init__()
        self.is_repeat = is_repeat
        if weight_mode not in ["normal", "uniform"]:
            raise ValueError("Weight_mode must be in ['normal','uniform']")
        self.weight_mode = weight_mode

    @staticmethod
    def random_normal(*size, mean=0.0, stddev=1.0, device=None, dtype=None):
        """Generate normal distributed tensor.

        Args:
            *size: Tensor dimensions
            mean (float): Distribution mean
            stddev (float): Distribution std deviation
            device: Output device
            dtype: Output data type

        Returns:
            torch.Tensor: Sampled values
        """
        return torch.randn(*size, dtype=dtype, device=device) * stddev + mean

    @staticmethod
    def random_uniform(*size, minval=0.0, max_val=1.0, device=None, dtype=None):
        """Generate uniform distributed tensor.

        Args:
            *size: Tensor dimensions
            minval (float): Minimum value
            max_val (float): Maximum value
            device: Output device
            dtype: Output data type

        Returns:
            torch.Tensor: Sampled values
        """
        return (max_val - minval) * torch.rand(*size, dtype=dtype, device=device) + minval

    def forward(self, image):
        """Apply color shift transformation.

        Args:
            image (torch.Tensor): Input image tensor (NCHW format)

        Returns:
            torch.Tensor: Transformed image tensor
        """
        N, C, H, W = image.shape
        dtype = image.dtype
        device = image.device
        r, g, b = torch.chunk(image, chunks=3, dim=1)

        # Generate channel weights
        if self.weight_mode == "normal":
            r_weight = self.random_normal(
                N, 1, 1, 1, mean=0.299, stddev=0.1, dtype=dtype, device=device
            )
            g_weight = self.random_normal(
                N, 1, 1, 1, mean=0.587, stddev=0.1, dtype=dtype, device=device
            )
            b_weight = self.random_normal(
                N, 1, 1, 1, mean=0.114, stddev=0.1, dtype=dtype, device=device
            )
        elif self.weight_mode == "uniform":
            r_weight = self.random_uniform(
                N, 1, 1, 1, minval=0.199, max_val=0.399, dtype=dtype, device=device
            )
            g_weight = self.random_uniform(
                N, 1, 1, 1, minval=0.487, max_val=0.687, dtype=dtype, device=device
            )
            b_weight = self.random_uniform(
                N, 1, 1, 1, minval=0.014, max_val=0.214, dtype=dtype, device=device
            )

        # Combine weighted channels
        output = (r_weight * r + g * g_weight + b * b_weight) / (
            r_weight + g_weight + b_weight + 1e-6
        )

        if self.is_repeat:
            output = output.repeat(1, 3, 1, 1)
        return output

    def get_num_channels(self):
        """Get number of output channels.

        Returns:
            int: 3 if repeating channels, else 1
        """
        return 3 if self.is_repeat else 1

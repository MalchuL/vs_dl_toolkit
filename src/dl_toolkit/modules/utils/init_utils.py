from torch import nn
from torch.nn import init


# TODO replace with code like https://github.com/open-mmlab/mmengine/blob/41fa84a9a922f19955ebb4265ec19ad10ee89991/mmengine/model/weight_init.py#L620
def init_weights(
        net: nn.Module, init_type: str = "kaiming_uniform", gain: float | None = None,
        nonlinearity="linear"
):
    def init_func(m):
        classname = m.__class__.__name__
        gain_value = gain
        if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                if gain_value is None:
                    raise ValueError("Must provide gain if init_type is 'uniform'.")
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain_value)
            elif init_type == "uniform":
                if gain_value is None:
                    raise ValueError("Must provide gain if init_type is 'uniform'.")
                # You can convert to it from normal distribution by gain = normal_gain * sqrt(12)
                init.uniform_(m.weight.data, a=-gain_value, b=gain_value)
            elif init_type == "xavier_normal":
                if gain_value is None:
                    gain_value = 1.0
                # TODO add weight clipping
                init.xavier_normal_(m.weight.data, gain=gain_value)
            elif init_type == "xavier_uniform":
                if gain_value is None:
                    gain_value = 1.0
                init.xavier_uniform_(m.weight.data, gain=gain_value)
            elif init_type == "kaiming_normal":
                if gain_value is None:
                    gain_value = 0.0
                # gain converts to = math.sqrt(2.0 / (1 + gain ** 2))
                # TODO add weight clipping
                init.kaiming_normal_(
                    m.weight.data, a=gain_value, mode="fan_in", nonlinearity=nonlinearity
                )
            elif init_type == "kaiming_uniform":
                if gain_value is None:
                    gain_value = 0.0
                # gain converts to = math.sqrt(2.0 / (1 + gain ** 2))
                init.kaiming_uniform_(
                    m.weight.data, a=gain_value, mode="fan_in", nonlinearity=nonlinearity
                )
            elif init_type == "zeros":
                init.zeros_(m.weight.data)
            elif init_type == "orthogonal":
                # https://hjweide.github.io/orthogonal-initialization-in-convolutional-layers
                if gain_value is None:
                    gain_value = 1.0
                init.orthogonal_(m.weight.data, gain=gain_value)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
                (classname.find("BatchNorm2d") != -1 or classname.find("GroupNorm") != -1)
                and hasattr(m, "weight")
                and m.weight is not None
        ):
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


def init_net(net: nn.Module, init_type: str, init_gain=None):
    if init_type is None:
        return net
    init_weights(net, init_type, gain=init_gain)

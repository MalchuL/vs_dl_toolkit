import torch

from dl_toolkit.modules.layers.conv.coord_conv import CoordConv


def test_coord_conv():
    coords = CoordConv()
    x = torch.randn(1, 3, 224, 224)
    other_x = torch.randn(4, 3, 128, 128)
    coords_x = coords(x)
    print(coords_x.shape)
    buffered_tensor = coords._buffered_tensor
    coords_other_x = coords(other_x)
    assert buffered_tensor is not coords._buffered_tensor

    print(coords_other_x.shape)
    buffered_tensor = coords._buffered_tensor
    coords_other_x = coords(other_x)
    assert buffered_tensor is coords._buffered_tensor
    print(coords_other_x.shape[2:], other_x.shape[2:])
    assert coords_other_x.shape[2:] == other_x.shape[2:]
    assert coords_other_x.shape[0] == other_x.shape[0]


def test_with_r():
    coords = CoordConv()
    coords_with_r= CoordConv(with_r=True)
    assert coords.channels == 2
    assert coords_with_r.channels == 3
    x = torch.randn(4, 3, 128, 128)
    out_x = coords(x)
    out_x_with_r = coords_with_r(x)
    assert out_x.shape[2:] == out_x_with_r.shape[2:]
    assert out_x_with_r.shape[1] == 3
    assert out_x_with_r.shape[0] == x.shape[0]

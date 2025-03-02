import numpy as np

from dl_toolkit.modules.layers.conv.representation.usm_sharp import gkern


def test_gkern():
    print(gkern(3, 1))
    print(gkern(5, 1))
    print(gkern(5, 2))
    print(gkern(5, 3))
    print(gkern(7, 3))
    print(gkern(7, 4))
    print(gkern(7, 1))

from enum import Enum


class TmpEnum(Enum):
    A = "1"
    B = "2"


def test_enum():
    instance = TmpEnum.A
    print(isinstance(instance, TmpEnum))
    print(isinstance(instance, str))

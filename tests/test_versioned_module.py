import pytest
import torch.nn as nn

from dl_toolkit.modules.losses import CharbonnierLoss

import logging

from dl_toolkit.utils.errors import ModuleVersionMismatchError

logging.basicConfig(level=logging.WARNING)

class RootModule(nn.Module):
    def __init__(self, version="1.0.0"):
        super().__init__()
        self.child = CharbonnierLoss()
        self.child.VERSION = version

def test_versioned_module():
    module1 = RootModule()
    module2 = RootModule()
    minor_change_version = "1.0.1"
    module3 = RootModule(minor_change_version)
    new_version = "1.1.1"
    module4 = RootModule(new_version)
    v2_version = "2.0.0"
    module5 = RootModule(v2_version)

    module1.load_state_dict(module2.state_dict())  # It's OK to load state_dict from the same module
    module1.load_state_dict(module3.state_dict())  # It's OK to load state_dict from the same module
    module1.load_state_dict(module4.state_dict())  # Only warning is printed
    with pytest.raises(ModuleVersionMismatchError):
        module1.load_state_dict(module5.state_dict())  # Error is raised


def test_non_versioned_module():
    CharbonnierLoss().load_state_dict(nn.Module().state_dict(), strict=False)
from typing import Any, Dict

import semver
import torch.nn as nn

from dl_toolkit.__version__ import __version__ as dl_toolkit_version
from dl_toolkit.utils.errors import ModuleVersionMismatchError
from dl_toolkit.utils.logging import logger


class VersionedModule(nn.Module):
    """
    Class for versioned modules. This should be used to check changes in the module. The version
    is stored in the state dict.
    You must update versions in next cases:
        bugfix version if new changes fixes bugs in the module.
        minors version if new changes fixes minor issues in the module or adds new
        features and methods.
        major version if new changes breaks compatibility of loading
        state dict, changes default parameters, fix module behavior according to paper or
        best practices.

    Examples:
        1. You fix module incorrect prints parameters. You refactor module for readability.
           In this case you should change bugfix version.
        2. You add support several behaviours and add switch between them in the module
           constructor (by default it's still original behaviour and last argument). In this
           case you should change minors version.
        3. In forward method you add addition of output by small epsilon. This change is small
           but you must change major version.

    In simple cases if you change anything that changes outputs you should change major version.
    It's applies to forward, initializations, changing default behaviour and so on.
    This logic is based on awareness to avoid incorrect module loading when package is
    updated when model was trained on previous version.

    """

    VERSION: str = "1.0.0"
    __VERSION_KEY = "__module_version__"
    __TOOLKIT_VERSION_KEY = "__toolkit_version__"

    def __init__(self):
        super().__init__()

    def get_extra_state(self) -> Dict[str, Any]:
        return {self.__VERSION_KEY: self.VERSION, self.__TOOLKIT_VERSION_KEY: dl_toolkit_version}

    def set_extra_state(self, state: Dict[str, Any]):
        """
        Called on load_state_dict
        :param state: dict
        :return:
        """
        if self.__VERSION_KEY in state:
            state_version = semver.Version.parse(state[self.__VERSION_KEY])
            module_version = semver.Version.parse(self.VERSION)
            # We should check major version
            if state_version.major != module_version.major:
                raise ModuleVersionMismatchError(
                    str(self),
                    module_version=self.VERSION,
                    state_dict_version=state[self.__VERSION_KEY],
                )
            elif state_version.minor != module_version.minor:
                logger.warning(
                    f"Minor version mismatch for {self}, {self.VERSION} != "
                    f"{state[self.__VERSION_KEY]}. It's recommended to use the "
                    f"same minor version, but must be okay"
                )
            # Update version to avoid warnings in the future
            self.VERSION = state[self.__VERSION_KEY]

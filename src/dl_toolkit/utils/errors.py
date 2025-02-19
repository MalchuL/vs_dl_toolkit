class DLToolkitError(Exception):
    pass


class ModuleVersionMismatchError(DLToolkitError):
    def __init__(self, module_info: str, module_version: str, state_dict_version: str):
        super().__init__(
            f"Module {module_info} version {module_version} "
            f"does not match state dict version {state_dict_version}"
        )

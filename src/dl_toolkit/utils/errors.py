class DLToolkitError(Exception):
    pass


class ModuleVersionMismatchError(DLToolkitError):
    def __init__(self, module_info: str, module_version: str, state_dict_version: str):
        super().__init__(
            f"Module {module_info} version {module_version} "
            f"does not match state dict version {state_dict_version}"
        )


class InvalidVersionError(DLToolkitError):
    def __init__(self, module_info: str, version: str):
        super().__init__(f"Invalid version {version} in {module_info}, it should be in format "
                         f"MAJOR.MINOR.PATCH")
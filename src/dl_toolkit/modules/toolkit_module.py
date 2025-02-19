from .versioned_module import VersionedModule


class ToolkitModule(VersionedModule):
    """
    A module that is part of the toolkit. Used by the toolkit.
    This module represent features and can only inherit from interfaces and classes.
    I recommend not to use this module in your classes.
    """

    pass

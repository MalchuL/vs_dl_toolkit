from typing import Type, Dict
import fnmatch
from torch import nn


def get_module_by_pattern(module: nn.Module,
                          access_string: str, 
                          match_each_dot: bool = False) -> Dict[str, nn.Module]:
    """Retrieve modules nested in another by an access string with wildcard patterns.

    This function supports Unix shell-style wildcards (e.g., *, ?) to match multiple modules.
    It uses PyTorch's named_modules() to traverse the hierarchy, ensuring compatibility with
    all container types (Sequential, ModuleList, etc.).

    Args:
        module (nn.Module): The root module.
        access_string (str): The access path with optional wildcards.
        match_each_dot (bool): Whether to match each dot-separated component individually.
            For example, *.bn will match all batchnorms only in FIRST layer.
            For other you must user *.*.bn for second layers batchnorms and so on.

    Returns:
        Dict[str, nn.Module]: Dictionary of matched modules.

    Examples:
        >>> model = nn.Sequential(nn.Conv2d(3, 6, 3), nn.ReLU())
        >>> get_module_by_pattern(model, '0')  # Matches Conv2d
        {'0': Conv2d(...)}
        >>> get_module_by_pattern(model, '*')  # Matches all layers
        {'0': Conv2d(...), '1': ReLU(...)}
    """

    matched_modules = {}
    if match_each_dot:
        target_pattern = access_string.split('.')

        for name, mod in module.named_modules(prefix=''):
            path = name.split('.') if name else []
            if len(path) != len(target_pattern):
                continue
            if all(fnmatch.fnmatch(p, t) for p, t in zip(path, target_pattern)):
                key = '.'.join(path)
                assert key not in matched_modules, f"Key {key} already exists"
                matched_modules[key] = mod
    else:
        for name, mod in module.named_modules(prefix=''):
            if fnmatch.fnmatch(name, access_string):
                assert name not in matched_modules, f"Key {name} already exists"
                matched_modules[name] = mod

    return matched_modules


def get_modules_by_type(module: nn.Module,
                        module_type: Type[nn.Module]) -> Dict[str, nn.Module]:
    """Retrieve all submodules of a specific type.

    Args:
        module (nn.Module): The root module.
        module_type (Type[nn.Module]): The type of modules to search for (e.g., nn.Conv2d).

    Returns:
        Dict[str, nn.Module]: Dictionary of matched modules.
    """
    if not isinstance(module, nn.Module):
        raise ValueError("Expected nn.Module, got " + str(type(module)))
    return {name: mod for name, mod in module.named_modules() if isinstance(mod, module_type)}


def get_modules_by_type_name(module: nn.Module,
                             type_name: str) -> Dict[str, nn.Module]:
    """Retrieve all submodules with class names matching the given pattern.

    Args:
        module (nn.Module): The root module.
        type_name (str): The class name pattern with wildcards (e.g., "Conv*").

    Returns:
        Dict[str, nn.Module]: Dictionary of matched modules.
    """
    if not isinstance(module, nn.Module):
        raise ValueError("Expected nn.Module, got " + str(type(module)))
    matched = {}
    for name, mod in module.named_modules():
        if fnmatch.fnmatch(mod.__class__.__name__, type_name):
            matched[name] = mod
    return matched

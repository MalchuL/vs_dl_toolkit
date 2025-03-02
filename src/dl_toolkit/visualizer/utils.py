# Copyright (c) OpenMMLab. All rights reserved.
import abc
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type, Union

import numpy as np

if TYPE_CHECKING:
    from matplotlib.backends.backend_agg import FigureCanvasAgg


def is_seq_of(seq: Any, expected_type: Union[Type, tuple], seq_type: Type = None) -> bool:
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type or tuple): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.

    Examples:
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def value2list(
    value: Any, valid_type: Union[Type, Tuple[Type, ...]], expand_dim: int
) -> List[Any]:
    """If the type of ``value`` is ``valid_type``, convert the value to list and expand to ``expand_dim``.

    Args:
        value (Any): value.
        valid_type (Union[Type, Tuple[Type, ...]): valid type.
        expand_dim (int): expand dim.

    Returns:
        List[Any]: value.
    """
    if isinstance(value, valid_type):
        value = [value] * expand_dim
    return value


def check_type(name: str, value: Any, valid_type: Union[Type, Tuple[Type, ...]]) -> None:
    """Check whether the type of value is in ``valid_type``.

    Args:
        name (str): value name.
        value (Any): value.
        valid_type (Type, Tuple[Type, ...]): expected type.
    """
    if not isinstance(value, valid_type):
        raise TypeError(f"`{name}` should be {valid_type} " f" but got {type(value)}")


def check_length(name: str, value: Any, valid_length: int) -> None:
    """If type of the ``value`` is list, check whether its length is equal with or greater than ``valid_length``.

    Args:
        name (str): value name.
        value (Any): value.
        valid_length (int): expected length.
    """
    if isinstance(value, list):
        if len(value) < valid_length:
            raise AssertionError(
                f"The length of {name} must equal with or "
                f"greater than {valid_length}, but got {len(value)}"
            )


def check_type_and_length(
    name: str, value: Any, valid_type: Union[Type, Tuple[Type, ...]], valid_length: int
) -> None:
    """Check whether the type of value is in ``valid_type``. If type of the ``value`` is list, check whether its length
    is equal with or greater than ``valid_length``.

    Args:
        value (Any): value.
        legal_type (Type, Tuple[Type, ...]): legal type.
        valid_length (int): expected length.

    Returns:
        List[Any]: value.
    """
    check_type(name, value, valid_type)
    check_length(name, value, valid_length)


def color_val_matplotlib(
    colors: Union[str, tuple, List[Union[str, tuple]]],
) -> Union[str, tuple, List[Union[str, tuple]]]:
    """Convert various input in RGB order to normalized RGB matplotlib color
    tuples,
    Args:
        colors (Union[str, tuple, List[Union[str, tuple]]]): Color inputs
    Returns:
        Union[str, tuple, List[Union[str, tuple]]]: A tuple of 3 normalized
        floats indicating RGB channels.
    """
    if isinstance(colors, str):
        return colors
    elif isinstance(colors, tuple):
        assert len(colors) == 3
        for channel in colors:
            assert 0 <= channel <= 255
        colors = [channel / 255 for channel in colors]
        return tuple(colors)
    elif isinstance(colors, list):
        colors = [
            color_val_matplotlib(color)  # type:ignore
            for color in colors
        ]
        return colors
    else:
        raise TypeError(f"Invalid type for color: {type(colors)}")


def color_str2rgb(color: str) -> tuple:
    """Convert Matplotlib str color to an RGB color which range is 0 to 255, silently dropping the alpha channel.

    Args:
        color (str): Matplotlib color.

    Returns:
        tuple: RGB color.
    """
    import matplotlib

    rgb_color: tuple = matplotlib.colors.to_rgb(color)
    rgb_color = tuple(int(c * 255) for c in rgb_color)
    return rgb_color


def wait_continue(figure, timeout: float = 0, continue_key: str = " ") -> int:
    """Show the image and wait for the user's input.

    This implementation refers to
    https://github.com/matplotlib/matplotlib/blob/v3.5.x/lib/matplotlib/_blocking_input.py

    Args:
        timeout (float): If positive, continue after ``timeout`` seconds.
            Defaults to 0.
        continue_key (str): The key for users to continue. Defaults to
            the space key.

    Returns:
        int: If zero, means time out or the user pressed ``continue_key``,
            and if one, means the user closed the show figure.
    """  # noqa: E501
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import CloseEvent

    is_inline = "inline" in plt.get_backend()
    if is_inline:
        # If use inline backend, interactive input and timeout is no use.
        return 0

    if figure.canvas.manager:  # type: ignore
        # Ensure that the figure is shown
        figure.show()  # type: ignore

    while True:
        # Connect the events to the handler function call.
        event = None

        def handler(ev):
            # Set external event variable
            nonlocal event
            # Qt backend may fire two events at the same time,
            # use a condition to avoid missing close event.
            event = ev if not isinstance(event, CloseEvent) else event
            figure.canvas.stop_event_loop()

        cids = [figure.canvas.mpl_connect(name, handler) for name in ("key_press_event", "close_event")]  # type: ignore

        try:
            figure.canvas.start_event_loop(timeout)  # type: ignore
        finally:  # Run even on exception like ctrl-c.
            # Disconnect the callbacks.
            for cid in cids:
                figure.canvas.mpl_disconnect(cid)  # type: ignore

        if isinstance(event, CloseEvent):
            return 1  # Quit for close.
        elif event is None or event.key == continue_key:
            return 0  # Quit for continue.


def img_from_canvas(canvas: "FigureCanvasAgg") -> np.ndarray:
    """Get RGB image from ``FigureCanvasAgg``.

    Args:
        canvas (FigureCanvasAgg): The canvas to get image.

    Returns:
        np.ndarray: the output of image in RGB.
    """  # noqa: E501
    s, (width, height) = canvas.print_to_buffer()
    buffer = np.frombuffer(s, dtype="uint8")
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    return rgb.astype("uint8")

"""
Contains utility functions for working with PRNG seeds and PyTorch devices.
"""
from dataclasses import dataclass
from datetime import datetime
from math import prod
from os import urandom
from pathlib import Path
from sys import byteorder

from torch import device
from torch.backends.mps import is_available as is_mps_available
from torch.cuda import is_available as is_cuda_available
from torch.nn import Module
from tqdm import tqdm


def choose_device(override: int | str | device | None = None) -> device:
    """
    Finds the most advanced computing device available on the current platform.

    The order of precedence is: CUDA, MPS, and default CPU.  This order is
    followed if `override` is empty.

    This function is adapted from Martin Mullang's blogpost "Simplifying PyTorch
    Device Selection" at: https://mctm.web.id/blog/2024/PyTorchGPUSelect/.

    Args:
        override: The device identifier to use regardless; optional.

    Returns:
        A PyTorch device.
    """
    if not override:
        if is_cuda_available():
            return device("cuda")
        elif is_mps_available():
            return device("mps")
        else:
            return device("cpu")

    return device(override)


def combine_dicts(original: dict[str, object],
                  updates: dict[str, object]) -> dict[str, object]:
    """
    Combines the keys and values of one dictionary into another.

    Args:
        original: A dictionary to modify.
        updates: A dictionary of keys and values to update.

    Returns:
        A combined dictionary.
    """
    for key, value in updates.items():
        if isinstance(value, dict):
            original_value = original.get(key, {})

            if isinstance(original_value, dict):
                combine_dicts(original_value, value)
            else:
                original[key] = value
        else:
            original[key] = value

    return original


def count_parameters(model: Module) -> int:
    """
    Computes the total number of trainable parameters in a model.

    Args:
        model: The model to search.

    Returns:
        The number of parameters as an integer.
    """
    return sum([prod(param.size()) for param in model.parameters()])


def random_bytes(bytes: int = 8) -> int:
    """
    Creates a single 64-bit random number using a system's `/dev/urandom`.

    Please note that while this function may technically raise an exception,
    this project assumes that it will not.

    Args:
        bytes: The number of random bytes to generate.  This should almost
        always be 64 bits (8 bytes).

    Returns:
        A random integer.
    """
    return int.from_bytes(urandom(bytes), byteorder)


def unique_name(model_name: str, timestamp: datetime | None = None,
                format: str | None = None) -> Path:
    """
    Creates a unique folder base name based on the current timestamp.

    Args:
        model_name: The name of the model.
        timestamp: The time to incorporate into the unique name.
        format: How to incorporate the name and timestamp information into a
        unique signature.

    Returns:
        A unique name as a standard string.
    """
    if not timestamp:
        timestamp = datetime.now()

    if not format:
        format = "{}-%d%m%Y-%H%M"

    return timestamp.strftime(format.format(model_name))


@dataclass(slots=True)
class Stopwatch:
    """
    A simple class to compute elapsed time for processes that perform
    significant work.
    """

    _start: datetime
    """The start time."""

    _end: datetime
    """The stop time."""

    def __init__(self):
        self._start = None
        self._end = None

    def start(self):
        """
        Assigns the current time as the start time.
        """
        self._start = datetime.now()

    def stop(self):
        """
        Assigns current time as the end time.
        """
        self._end = datetime.now()

    def elapsed(self, format: str | None = None) -> str:
        """
        Computes the elapsed time and formats it as appropriate for visual
        display.

        Args:
            format: The output format to use.

        Returns:
            The elapsed time as a formatted string.
        """
        if not format:
            format = "{} hours, {} minutes, and {} seconds"

        elapsed = self._end - self._start

        hours = elapsed.seconds // 3600
        minutes = (elapsed.seconds % 3600) // 60
        seconds = elapsed.seconds % 60

        if hours == 1:
            format = format.replace("hours", "hour")

        if minutes == 1:
            format = format.replace("minutes", "minute")

        if seconds == 1:
            format = format.replace("seconds", "second")

        return format.format(hours, minutes, seconds)


class TqdmSink:
    """
    Wraps :meth:`tqdm.write` in an object Loguru can use to send log messages
    and subsequently display them.
    """

    __slots__ = ()

    def write(self, message: str):
        """
        Writes a formatted message to the console using TQDM's facilities to
        display it above any active progress bars.

        Args:
            message: The message to display.
        """
        tqdm.write(message, end="")

    def stop(self):
        """
        Intended for stopping continously active resources.

        For this class, no action is needed.
        """
        pass



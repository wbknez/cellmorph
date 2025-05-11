"""
Contains common fixtures and utilities for unit testing.
"""
from pathlib import Path
from string import ascii_letters, digits

from numpy import int32, int64, uint8
from numpy.random import Generator, SeedSequence, default_rng
from PIL import Image as ImageFactory
from PIL.Image import Image
from pytest import fixture
from torch import Tensor, rand


@fixture(scope="function")
def rng(seed: int | int32 | int64 | None = None) -> Generator:
    """
    Creates a new pseudo-random number generator using a particular seed.

    Args:
        seed: An entropy value; default is `None`.

    Returns:
        A new pseudo-random number generator.
    """
    return default_rng(SeedSequence(seed))


@fixture(scope="function")
def random_path(rng: Generator) -> Path:
    """
    Creates a random path from a random combination of numbers and letters.

    Args:
        rng: A pseudo-random number generator.

    Returns:
        A random path.
    """
    choices = ascii_letters + digits
    chosen = [choices[i] for i in rng.choice(len(choices), 20)]

    return Path("".join(chosen))


@fixture(scope="function")
def img(width: int, height: int, color_channels: int, rng: Generator) -> Image:
    """
    Creates an :class:`Image` filled with random pixel data.

    Args:
        width: The image width in cells.
        height: The image height in cells.
        color_channels: The number of color channels.
        rng: A pseudo-random number generator.

    Returns:
        A random image.
    """
    mode = "RGBA" if color_channels == 4 else "RGB"

    return ImageFactory.fromarray(rng.integers(
        0, 255, (height, width, color_channels), dtype=uint8
    ), mode=mode)


@fixture(scope="function")
def x(sample_count: int, state_channels: int, height: int,
      width: int) -> Tensor:
    """
    Creates a sample :class:`Tensor` filled with random floating-point values.

    Args:
        sample_count: The number of samples to create.
        state_channels: The number of state channels.
        height: The height.
        width: The width.

    Returns:
        A random tensor.
    """
    return rand((sample_count, state_channels, height, width)).float()


@fixture(scope="function")
def target(target_count, height: int, width: int) -> Tensor:
    """
    Creates a target :class:`Tensor` filled with random floating-point values.

    Args:
        target_count: The number of targets to create.
        state_channels: The number of state channels.
        height: The height.
        width: The width.

    Returns:
        A random tensor.
    """
    return rand((target_count, 4, height, width)).float()

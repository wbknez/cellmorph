"""
Contains common fixtures and utilities for unit testing.
"""
from typing import Optional, Union

from numpy import int32, int64
from numpy.random import Generator, SeedSequence, default_rng
from pytest import fixture


@fixture(scope="function")
def rng(seed: Optional[Union[int, int32, int64]] = None) -> Generator:
    """
    Creates a new pseudo-random number generator using a particular seed.

    Args:
        seed: An entropy value; default is `None`.

    Returns:
        A new pseudo-random number generator.
    """
    return default_rng(SeedSequence(seed))

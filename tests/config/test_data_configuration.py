"""
Ensures that data configuration initialization works correctly.
"""
from pathlib import Path
from string import ascii_letters, digits

from numpy.random import Generator
from pytest import fixture, mark, raises

from cellmorph.config import DataConfiguration
from cellmorph.image import Dimension


@fixture(scope="function")
def cache_dir(rng: Generator) -> Path:
    choices = ascii_letters + digits
    chosen = [choices[i] for i in rng.choice(len(choices), 20)]

    return "".join(chosen)


class TestDataConfiguration:
    """
    Test suite for :class:`DataConfiguration`.
    """

    @mark.parametrize('sample_count', [ (0), (-1) ])
    def test_data_config_raises_if_sample_count_is_not_positive(
        self, sample_count: int, cache_dir: Path
    ):
        with raises(ValueError):
            DataConfiguration("wishjs", sample_count, Dimension(40, 40), 16,
                              True, cache_dir)

    @mark.parametrize('target', [ None, "" ])
    def test_data_config_raises_if_target_is_not_valid(
        self, target: str, cache_dir: Path
    ):
        with raises(ValueError):
            DataConfiguration(target, 1024, Dimension(40, 40), 16, True,
                              cache_dir)

    @mark.parametrize('cache_dir_t', [ None, "" ])
    def test_data_config_raises_if_target_is_not_valid(
        self, cache_dir_t: Path
    ):
        with raises(ValueError):
            DataConfiguration("sksks", 1024, Dimension(40, 40), 16, True,
                              cache_dir_t)

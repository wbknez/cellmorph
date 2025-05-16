"""
Ensures that loss collection and serialization work as intended.
"""
from pathlib import Path

from numpy import genfromtxt
from numpy.random import Generator
from pytest import mark
from torch import rand

from cellmorph.training import Losses


class TestLosses:
    """
    Test suite for :class:`Losses`.
    """

    @mark.parametrize('count', [
        (5),
        (50),
        (1024)
    ])
    def test_append_adds_individual_values_one_at_a_list(
        self, count: int, rng: Generator
    ):
        values = rng.random(count).tolist()
        losses = Losses()

        for value in values:
            losses.append(value)

        assert losses._values == values

    @mark.parametrize('count, header', [
        (5, False),
        (5, True),
        (50, False),
        (50, True),
        (1024, False),
        (1024, True)
    ])
    def test_save_writes_values_in_single_column(
        self, count: int, header: bool, rng: Generator, tmpdir: Path
    ):
        loss_path = tmpdir / "losses.csv"
        skip_header = 1 if header else 0

        values = rng.random(count).tolist()
        Losses(values).save(loss_path, header)

        expected = Losses(values)
        result = Losses( genfromtxt(
            loss_path, skip_header=skip_header, delimiter=","
        ).tolist())

        assert result == expected

"""
Ensures that interval attributes obey value constraints.
"""
from pytest import mark, raises

from cellmorph.training import Interval


class TestInterval:
    """
    Test suite for :class:`Interval`.
    """

    @mark.parametrize('a, b', [ (4, 4), (55, 37), (55, 54) ])
    def test_interval_raises_if_a_is_greater_than_or_equal_to_b(
        self, a: int, b: int
    ):
        with raises(ValueError):
            Interval(a, b)

    @mark.parametrize('a, b', [ (0, 5), (-1, 66) ])
    def test_interval_raises_if_a_is_not_positive(self, a: int, b: int):
        with raises(ValueError):
            Interval(a, b)

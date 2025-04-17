"""

"""
from pytest import mark, raises

from cellmorph.training import Position

class TestPosition:
    """
    Test suite for :class:`Position`.
    """

    @mark.parametrize('x', [-1, -2])
    def test_position_raises_if_x_is_negative(self, x: int):
        with raises(ValueError):
            Position(x, 0)

    @mark.parametrize('y', [-1, -2])
    def test_position_raises_if_y_is_negative(self, y: int):
        with raises(ValueError):
            Position(0, y)

"""
Ensures that determining whether an image can be shrunk (thumbnail'd) works
correctly.
"""
from pytest import mark

from cellmorph.image import Dimension, can_thumbnail


class TestCanThumbnail:
    """
    Test suite for :meth:`can_thumbnail`.
    """

    @mark.parametrize('size, max_size', [
        (Dimension(40, 40), Dimension(41, 39)),
        (Dimension(30, 35), Dimension(31, 33)),
        (Dimension(36, 24), Dimension(37, 19))
    ])
    def test_can_thumbnail_returns_false_if_x_axis_is_larger(
        self, size: Dimension, max_size: Dimension
    ):
        assert not can_thumbnail(size, max_size)

    @mark.parametrize('size, max_size', [
        (Dimension(40, 40), Dimension(12, 41)),
        (Dimension(30, 35), Dimension(26, 36)),
        (Dimension(36, 24), Dimension(32, 25))
    ])
    def test_can_thumbnail_returns_false_if_y_axis_is_larger(
        self, size: Dimension, max_size: Dimension
    ):
        assert not can_thumbnail(size, max_size)

    @mark.parametrize('size, max_size', [
        (Dimension(40, 40), Dimension(40, 40)),
        (Dimension(30, 35), Dimension(30, 35)),
        (Dimension(36, 24), Dimension(36, 24))
    ])
    def test_can_thumbnail_returns_false_if_both_axes_are_equal(
        self, size: Dimension, max_size: Dimension
    ):
        assert not can_thumbnail(size, max_size)

    @mark.parametrize('size, max_size', [
        (Dimension(40, 40), Dimension(40, 39)),
        (Dimension(30, 35), Dimension(30, 29)),
        (Dimension(36, 24), Dimension(13, 22))
    ])
    def test_can_thumbnail_returns_true_if_max_size_less_than_size(
        self, size: Dimension, max_size: Dimension
    ):
        assert can_thumbnail(size, max_size)

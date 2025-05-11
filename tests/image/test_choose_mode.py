"""
Checks that mode determination from channel count works as expected.
"""
from pytest import mark, raises

from cellmorph.image import choose_mode


class TestChooseMode:
    """
    Test suite for :meth:`choose_mode`.
    """

    def test_choose_mode_raises_if_color_channels_are_out_of_bounds(self):
        with raises(ValueError):
            choose_mode(0)

        with raises(ValueError):
            choose_mode(5)

    @mark.parametrize('color_channels, expected', [
        (1, "L"),
        (2, "LA"),
        (3, "RGB"),
        (4, "RGBA")
    ])
    def test_choose_mode_chooses_mode_correctly(
        self, color_channels: int, expected: str
    ):
        result = choose_mode(color_channels)

        assert result == expected

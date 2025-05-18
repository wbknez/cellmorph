"""
Ensures that obtaining :class:`Dimension` values from string-based dictionary
structures works as expected.
"""
from pytest import mark, raises

from cellmorph.config import YAML, get_dimension
from cellmorph.data import Dimension


class TestGetDimension:
    """
    Test suite for :meth:`get_dimension`.
    """

    @mark.parametrize('yaml, expected', [
        ({ "value": { "width": 3, "height": 10 } }, Dimension(3, 10)),
        ({ "value": { "width": "30", "height": "19" } }, Dimension(30, 19)),
        ({ }, Dimension(1, 1))
    ])
    def test_get_dimension_accepts_int_and_strings_only(
        self, yaml: YAML, expected: int
    ):
        result = get_dimension(yaml, "value", Dimension(1, 1))

        assert isinstance(result, Dimension)
        assert result == expected

    @mark.parametrize('yaml', [
        { "value": { "width": 3.143245, "height": 42 } },
        { "value": { "width": 97, "height": "0x00a3f" } },
        { "value": [ 93, 102 ] }
    ])
    def test_get_dimension_raises_if_value_is_not_a_dimension(self, yaml: YAML):
        with raises(ValueError):
            get_dimension(yaml, "value", 0)

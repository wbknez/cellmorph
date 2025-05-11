"""
Ensures that obtaining :class:`Interval` values from string-based dictionary
structures works as expected.
"""
from pytest import mark, raises

from cellmorph.config import YAML, Interval, get_interval


class TestGetInterval:
    """
    Test suite for :meth:`get_dimension`.
    """

    @mark.parametrize('yaml, expected', [
        ({ "value": { "min": 3, "max": 10 } }, Interval(3, 10)),
        ({ "value": { "min": "19", "max": "30" } }, Interval(19, 30)),
        ({ }, Interval(1, 2))
    ])
    def test_get_dimension_accepts_int_and_strings_only(
        self, yaml: YAML, expected: int
    ):
        result = get_interval(yaml, "value", Interval(1, 2))

        assert isinstance(result, Interval)
        assert result == expected

    @mark.parametrize('yaml', [
        { "value": { "min": 3.143245, "max": 42 } },
        { "value": { "min": 97, "max": "0x00a3f" } },
        { "value": [ 93, 102 ] }
    ])
    def test_get_dimension_raises_if_value_is_not_a_dimension(self, yaml: YAML):
        with raises(ValueError):
            get_interval(yaml, "value", Interval(1, 2))

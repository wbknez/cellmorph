"""
Ensures that obtaining integer values from string-based dictionary structures
works as expected.
"""
from pytest import mark, raises

from cellmorph.config import YAML, get_int


class TestGetInt:
    """
    Test suite for :meth:`get_int`.
    """

    @mark.parametrize('yaml, expected', [
        ({ "value": 42 }, 42),
        ({ "value": "97" }, 97),
        ({ }, 0)
    ])
    def test_get_int_accepts_integers_and_strings_only(
        self, yaml: YAML, expected: int
    ):
        result = get_int(yaml, "value", 0)

        assert isinstance(result, int)
        assert result == expected

    @mark.parametrize('yaml', [
        { "value": "not an integer" },
        { "value": "0x00a3f" },
        { "value": 3.1312553 },
        { "value": [ 1, 2, 3 ] }
    ])
    def test_get_int_raises_if_value_is_not_an_integer(self, yaml: YAML):
        with raises(ValueError):
            get_int(yaml, "value", 0)

"""
Ensures that obtaining boolean values from string-based dictionary structures
works as expected.
"""
from pytest import mark, raises

from cellmorph.config import YAML, get_bool


class TestGetBool:
    """
    Test suite for :meth:`get_bool`.
    """

    @mark.parametrize('yaml, expected', [
        ({ "value": 1 }, True),
        ({ "value": 0 }, False),
        ({ "value": True }, True),
        ({ "value": False }, False),
        ({ "value": "True" }, True),
        ({ "value": "1" }, True),
        ({ "value": "False" }, False),
        ({ "value": "0" }, False),
        ({ }, False)
    ])
    def test_get_bool_accepts_booleans_integers_and_strings_only(
        self, yaml: YAML, expected: bool
    ):
        result = get_bool(yaml, "value", False)

        assert isinstance(result, bool)
        assert result == expected

    @mark.parametrize('yaml', [
        { "value": "not an boolean" },
        { "value": "0x00a3f" },
        { "value": 3.1312553 },
        { "value": [ 1, 2, 3 ] },
    ])
    def test_get_bool_raises_if_value_is_not_a_boolean(self, yaml: YAML):
        with raises(ValueError):
            get_bool(yaml, "value", 0)

"""
Ensures that obtaining string values from string-based dictionary structures
works as expected.
"""
from pytest import mark, raises

from cellmorph.config import YAML, get_str


class TestGetStr:
    """
    Test suite for :meth:`get_str`.
    """

    @mark.parametrize('yaml, expected', [
        ({ "value": "Some cool string" }, "Some cool string"),
        ({ }, "something")
    ])
    def test_get_str_accepts_strings_only(
        self, yaml: YAML, expected: str
    ):
        result = get_str(yaml, "value", "something")

        assert isinstance(result, str)
        assert result == expected

    @mark.parametrize('yaml', [
        { "value": [ 1, 2, 3 ] },
        { "value": { "a": 4, "b": 2 } }
    ])
    def test_get_str_raises_if_value_is_not_a_string(self, yaml: YAML):
        with raises(ValueError):
            get_str(yaml, "value", 0)

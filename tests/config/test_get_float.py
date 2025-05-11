"""
Ensures that obtaining float values from string-based dictionary structures
works as expected.
"""
from pytest import mark, raises

from cellmorph.config import YAML, get_float


class TestGetFloat:
    """
    Test suite for :meth:`get_float`.
    """

    @mark.parametrize('yaml, expected', [
        ({ "value": 42 }, 42.0),
        ({ "value": "97" }, 97.0),
        ({ "value": 3.1417 }, 3.1417),
        ({ "value": "4.8528" }, 4.8528),
        ({ }, 0.001)
    ])
    def test_get_float_accepts_floats_ints_and_strings_only(
        self, yaml: YAML, expected: float
    ):
        result = get_float(yaml, "value", 0.001)

        assert isinstance(result, float)
        assert result == expected
    
    @mark.parametrize('yaml', [
        { "value": "not an float" },
        { "value": "0x00a3f" },
        { "value": [ 1, 2, 3 ] }
    ])
    def test_get_float_raises_if_value_is_not_a_float(self, yaml: YAML):
        with raises(ValueError):
            get_float(yaml, "value", 0)

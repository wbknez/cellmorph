"""
Ensures that combining two dictionaries works as expected.
"""
from pytest import mark

from cellmorph.utils import combine_dicts


class TestCombineDicts:
    """
    Test suite for :meth:`combined_dicts`.
    """

    @mark.parametrize('original, updates, expected', [
        (
            { "a": 3, "b": { "d": 2, "e": 1 }, "c": "hullo" },
            { "b": { "d": 3.141719 }, "f": [ 1, 2, 3, 4, 5 ] },
            { "a": 3, "b": { "d": 3.141719, "e": 1 }, "c": "hullo",
              "f": [ 1, 2, 3, 4, 5 ] },
        )
    ])
    def test_combine_dicts_combines_dictionaries_correctly(
        self, original: dict[str, object], updates: dict[str, object],
        expected: dict[str, object]
    ):
        result = combine_dicts(original, updates)

        assert result == expected

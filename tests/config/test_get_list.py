"""
Ensures that obtaining typed :class:`list` values from string-based dictionary
structures works as expected.
"""
from pathlib import Path

from pytest import mark, raises

from cellmorph.config import YAML, Value, get_list


class TestGetList:
    """
    Test suite for :meth:`get_list`.
    """

    @mark.parametrize('yaml, ctype, expected', [
        ({ "value": [ 1, 2, 3, 4 ] }, int, [1, 2, 3, 4]),
        ({ "value": [ "1", "2", "3", "4" ] }, int, [1, 2, 3, 4]),
        (
            { "value": [ 1.2, 9.2, 3.1417, 4.01 ] },
            float,
            [1.2, 9.2, 3.1417, 4.01]
        ),
        (
            { "value": [ "1.2", "9.2", "3.1417", "4.01" ] },
            float,
            [1.2, 9.2, 3.1417, 4.01]
        ),
        (
            { "value": [ Path("hi"), Path("some/dir"), Path("iron/within") ] },
            Path,
            [ Path("hi"), Path("some/dir"), Path("iron/within") ]
        ),
        (
            { "value": [ "hi", "some/dir", "iron/within" ] },
            Path,
            [ Path("hi"), Path("some/dir"), Path("iron/within") ]
        ),
        ({ "value": [ "1", "2", "3", "4" ] }, str, ["1", "2", "3", "4"]),
    ])
    def test_get_list_accepts_only_types_and_strings(
        self, yaml: YAML, ctype: Value, expected: Value
    ):
        result = get_list(yaml, "value", ctype, [])

        assert isinstance(result, list)
        assert result == expected

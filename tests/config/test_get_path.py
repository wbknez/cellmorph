"""
Ensures that obtaining :class:`Path` values from string-based dictionary
structures works as expected.
"""
from pathlib import Path

from pytest import mark, raises

from cellmorph.config import YAML, get_path


class TestGetPath:
    """
    Test suite for :meth:`get_path`.
    """

    @mark.parametrize('yaml, expected', [
        ({ "value": Path("test_path") }, Path("test_path")),
        ({ "value": "some/path" }, Path("some/path")),
        ({ }, Path("./"))
    ])
    def test_get_path_accepts_paths_and_strings_only(
        self, yaml: YAML, expected: Path
    ):
        result = get_path(yaml, "value", Path("./"))

        assert isinstance(result, Path)
        assert result == expected
    
    @mark.parametrize('yaml', [
        { "value": 42 },
        { "value": 3.1417 },
        { "value": [ 1, 2, 3 ] }
    ])
    def test_get_path_raises_if_value_is_not_a_path(self, yaml: YAML):
        with raises(ValueError):
            get_path(yaml, "value", 0)

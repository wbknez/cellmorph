"""
Ensures that stripping the prefix from dictionary keys works as expected.
"""
from pytest import mark
from torch import rand

from cellmorph.utils import strip_key_prefix


WEIGHT_0 = rand((50,)).numpy().tolist()
WEIGHT_1 = rand((100,)).numpy().tolist()
WEIGHT_2 = rand((12, 17, 96)).numpy().tolist()


class TestStripKeyPrefix:
    """
    Test suite for :meth:`strip_key_prefix`.
    """

    @mark.parametrize('mapped, prefix, expected', [
        (
            { "_prefix.key0": 42, "_prefix.key1": 'Hi!', "key2": 3.1419 },
            "_prefix.",
            { "key0": 42, "key1": 'Hi!', "key2": 3.1419 }
        ),
        (
            {
                "_orig_mod.weight0": WEIGHT_0,
                "_orig_mod.weight1": WEIGHT_1,
                "_orig_mod.weight2": WEIGHT_2
            },
            "_orig_mod.",
            {
                "weight0": WEIGHT_0,
                "weight1": WEIGHT_1,
                "weight2": WEIGHT_2
            }
        )
    ])
    def test_strip_key_prefix_strips_prefix_from_keys(
        self, mapped: dict[str, object], prefix: str,
        expected: dict[str, object]
    ):
        result = strip_key_prefix(mapped, prefix)

        assert set(result.keys()) == set(expected.keys())
        assert result == expected

"""
Ensures that emoji conversion from a hexadecimal code works correctly.
"""
from pytest import mark

from cellmorph.emoji import CommonEmojis, to_emoji


class TestToEmoji:
    """
    Test suite for :meth:`to_emoji`.
    """

    @mark.parametrize('code, expected', [
        (CommonEmojis.GECKO, "🦎"),
        (CommonEmojis.SMILEY, "😀"),
        (CommonEmojis.EXPLOSION, "💥"),
        (CommonEmojis.EYE, "👁"),
        (CommonEmojis.FISH, "🐠"),
        (CommonEmojis.BUTTERFLY, "🦋"),
        (CommonEmojis.LADYBUG, "🐞"),
        (CommonEmojis.SPIDERWEB, "🕸"),
        (CommonEmojis.PRETZEL, "🥨"),
        (CommonEmojis.CHRISTMAS_TREE, "🎄")
    ])
    def test_to_emoji_converts_as_expected(self, code: str, expected: str):
        assert to_emoji(code) == expected

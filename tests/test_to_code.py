"""
Ensures that converting an emoji character into a hex string is correct.
"""
from pytest import mark, raises

from cellmorph.emoji import CommonEmojis, to_code


class TestToCode:
    """
    Test suite for :meth:`from_code`.
    """

    def test_to_code_raises_if_input_is_not_an_emoji(self):
        with raises(ValueError):
            to_code("hiadiohs")

        with raises(ValueError):
            to_code(CommonEmojis.LADYBUG)

    @mark.parametrize('emoji, expected', [
        ('🦎', CommonEmojis.GECKO), 
        ('😀', CommonEmojis.SMILEY),
        ('💥', CommonEmojis.EXPLOSION),
        ('👁', CommonEmojis.EYE),
        ('🐠', CommonEmojis.FISH),
        ('🦋', CommonEmojis.BUTTERFLY),
        ('🐞', CommonEmojis.LADYBUG),
        ('🕸', CommonEmojis.SPIDERWEB),
        ('🥨', CommonEmojis.PRETZEL),
        ('🎄', CommonEmojis.CHRISTMAS_TREE)
    ])
    def test_to_code_converts_to_hex(self, emoji: str, expected):
        result = to_code(emoji)

        assert result == expected

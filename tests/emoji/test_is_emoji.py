"""
Ensures that emoji detection works as expected.
"""
from pytest import mark

from cellmorph.emoji import CommonEmojis, is_emoji, to_emoji


CODES = [to_emoji(emoji) for emoji in CommonEmojis]


class TestIsEmoji:
    """
    Test suite for :meth:`is_emoji`.
    """

    @mark.parametrize('emoji', [ "0x001293", "0x099918239a", "1", "0" ])
    def test_is_emoji_returns_false_if_not_an_emoji(self, emoji: str):
        assert not is_emoji(emoji)

    @mark.parametrize('emoji', CODES)
    def test_is_emoji_returns_true_if_is_an_emoji(self, emoji: str):
        assert is_emoji(emoji)

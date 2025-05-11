"""
Ensures that emoji URL creation (Noto font(s)) works as expected.
"""
from pytest import mark, raises

from cellmorph.emoji import CommonEmojis, make_url


class TestMakeUrl:
    """
    Test suite for :meth:`make_url`.
    """

    def test_make_url_raises_if_image_size_is_out_of_bounds(self):
        with raises(ValueError):
            make_url(CommonEmojis.BUTTERFLY, 13)

    @mark.parametrize('emoji_code, image_size', [
        (CommonEmojis.GECKO, 512),
        (CommonEmojis.BUTTERFLY, 128),
        (CommonEmojis.SPIDERWEB, 72)
    ])
    def test_make_url_produces_correct_url_to_noto_emojis(
        self, emoji_code: str, image_size: int
    ):
        expected = "https://github.com/googlefonts/noto-emoji/"
        expected += f"blob/main/png/{image_size}/emoji_u{emoji_code}"
        expected += ".png?raw=true"

        result = make_url(emoji_code, image_size)

        assert result == expected

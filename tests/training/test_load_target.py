"""
Ensures that loading targets from a file path or emoji symbol works as expected.
"""
from pathlib import Path

from numpy import array, array_equal, uint8
from PIL.Image import Image
from pytest import mark

from cellmorph.emoji import CommonEmojis, EmojiSizes, fetch_emoji, to_code
from cellmorph.training import load_target


def image_equal(a: Image, b: Image) -> bool:
    x = array(a, dtype="uint8")
    y = array(b, dtype="uint8")

    return array_equal(x, y)


class TestLoadTarget:
    """
    Test suite for :meth:`load_target`.
    """

    @mark.parametrize('width, height, color_channels', [
        (40, 40, 4),
        (72, 72, 3)
    ])
    def test_load_target_with_image_loads_correctly_with_no_cache(
        self, img: Image, tmpdir: Path
    ):
        img_path = tmpdir / "image.png"
        img.save(img_path)

        result = load_target(img_path)

        assert image_equal(result, img)

        result.close()
        img.close()

    @mark.parametrize('emoji, size', [
        (CommonEmojis.BUTTERFLY, EmojiSizes.i128),
        (CommonEmojis.GECKO, EmojiSizes.i72)
    ])
    def test_load_target_with_emoji_with_code(
        self, emoji: str, size: int
    ):
        expected = fetch_emoji(emoji, size)
        result = load_target(emoji, size)

        assert image_equal(result, expected)

        result.close()
        expected.close()

    @mark.parametrize('emoji, size', [
        ("🦎", EmojiSizes.i128),
        ("👁", EmojiSizes.i72)
    ])
    def test_load_target_with_raw_emoji(
        self, emoji: str, size: int
    ):
        expected = fetch_emoji(to_code(emoji), size)
        result = load_target(emoji, size)

        assert image_equal(result, expected)

        result.close()
        expected.close()

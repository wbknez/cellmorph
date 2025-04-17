"""

"""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from numpy import array, array_equal, uint8
from numpy.random import Generator
from PIL import Image as ImageFactory
from PIL.Image import Image
from pytest import fixture, mark
from requests import get

from cellmorph.emoji import CommonEmojis, fetch_emoji, make_url


@fixture(scope="function")
def img(width: int, height: int, rng: Generator) -> Image:
    return ImageFactory.fromarray(rng.integers(
        0, 255, (height, width, 4), dtype=uint8
    ), mode="RGBA")


class TestFetchEmoji:
    """
    Test suite for :meth:`fetch_emoji`.
    """

    @mark.filterwarnings("error")
    @mark.parametrize('emoji_code, width, height', [
        (CommonEmojis.BUTTERFLY, 128, 128),
        (CommonEmojis.SPIDERWEB, 40, 40)
    ])
    def test_fetch_emoji_with_cached_dir_and_cached_image(
        self, emoji_code: str, img: Image, tmp_path: Path
    ):
        cache_dir = tmp_path

        img.save(f"{cache_dir}/{emoji_code}.png")

        with fetch_emoji(emoji_code, cache_dir=cache_dir) as result:
            expected = img.copy()

            assert result.mode == expected.mode
            assert array_equal(array(result), array(expected))

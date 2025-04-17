"""
Ensures that image padding works as expected.
"""
from numpy import uint8
from numpy.random import Generator
from PIL import Image as ImageFactory
from PIL.Image import Image
from pytest import fixture, mark

from cellmorph.image import pad


@fixture(scope="function")
def img(rng: Generator, width: int, height: int) -> Image:
    arr = rng.integers(0, 255, (height, width, 4)).astype(uint8)

    return ImageFactory.fromarray(arr, mode="RGBA")


class TestPad:
    """
    Test suite for :meth:`pad`.
    """

    @mark.parametrize("width, height", [ (40, 40), (120, 97), (19, 108) ])
    def test_pad_returns_image_unmodified_if_padding_is_zero(self, img: Image):
        expected = img.copy()
        result = pad(img, 0)

        assert result == expected
        assert not result is expected

    @mark.parametrize("width, height, padding", [
        (40, 40, 16),
        (120, 97, 9),
        (19, 108, 88)
    ])
    def test_pad_returns_modified_image(self, padding: int, img: Image):
        width, height = img.size
        new_width, new_height = width + padding * 2, height + padding * 2

        expected = ImageFactory.new(img.mode, (new_width, new_height))
        expected.paste(img, (padding, padding))
        result = pad(img, padding)

        assert result == expected

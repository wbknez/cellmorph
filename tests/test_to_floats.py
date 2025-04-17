"""

"""
from numpy import array, array_equal, float32, uint8
from numpy.random import Generator
from PIL import Image as ImageFactory
from PIL.Image import Image
from pytest import fixture, mark

from cellmorph.image import to_floats


@fixture(scope="function")
def img(rng: Generator, width: int, height: int) -> Image:
    arr = rng.integers(0, 255, (height, width, 4)).astype(uint8)

    return ImageFactory.fromarray(arr, mode="RGBA")


class TestToFloats:
    """
    Test suite for :meth:`to_floats`.
    """

    @mark.parametrize("width, height, premultiply", [
        (40, 40, False),
        (40, 40, True),
        (120, 97, False),
        (120, 97, True),
        (19, 108, False),
        (19, 108, True)
    ])
    def test_to_floats_converts_to_rgba_array_from_image(
        self, img: Image, premultiply: bool
    ):
        expected = array(img, dtype=float32) / 255.0
        result = to_floats(img, premultiply)

        if premultiply:
            expected[..., :3] *= expected[..., 3:4]

        assert array_equal(result, expected)

    @mark.parametrize("width, height, premultiply", [
        (40, 40, False),
        (40, 40, True),
        (120, 97, False),
        (120, 97, True),
        (19, 108, False),
        (19, 108, True)
    ])
    def test_to_floats_converts_to_rgba_array_from_uint_array(
        self, img: Image, premultiply: bool
    ):
        expected = array(img, dtype=float32) / 255.0
        result = to_floats(array(img, dtype=uint8), premultiply)

        if premultiply:
            expected[..., :3] *= expected[..., 3:4]

        assert array_equal(result, expected)

    @mark.parametrize("width, height, premultiply", [
        (40, 40, False),
        (40, 40, True),
        (120, 97, False),
        (120, 97, True),
        (19, 108, False),
        (19, 108, True)
    ])
    def test_to_floats_converts_to_rgba_array_from_float_array(
        self, width: int, height: int, premultiply: bool, rng: Generator
    ):
        arr = rng.random((height, width, 4), dtype=float32)

        expected = arr
        result = to_floats(arr, premultiply)

        if premultiply:
            expected[..., :3] *= expected[..., 3:4]

        assert array_equal(result, expected)

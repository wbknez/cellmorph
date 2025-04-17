"""
Ensures that converting an image to a tensor works as expected.
"""
from numpy import uint8
from numpy.random import Generator
from PIL import Image as ImageFactory
from PIL.Image import Image
from pytest import fixture, mark
from torch import Tensor, equal, from_numpy

from cellmorph.image import to_floats, to_tensor


@fixture(scope="function")
def img(width: int, height: int, color_channels: int, rng: Generator) -> Image:
    arr = rng.integers(0, 255, (height, width, color_channels), dtype=uint8)
    mode = "RGBA" if color_channels == 4 else "RGB"

    return ImageFactory.fromarray(arr, mode)


class TestToTensor:
    """
    Test suite for :meth:`to_tensor`.
    """

    @mark.parametrize('width, height, color_channels', [
        (40, 40, 4),
        (72, 17, 3),
        (21, 29, 4)
    ])
    def test_to_tensor_converts_image(self, img: Image):
        expected = from_numpy(to_floats(img).transpose(2, 0, 1)).unsqueeze(0)
        result = to_tensor(img)

        assert equal(result, expected)

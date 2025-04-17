"""
Checks that preparing images as training input works correctly.
"""
from dataclasses import astuple

from numpy import uint8
from numpy.random import Generator
from PIL import Image as ImageFactory
from PIL.Image import Image
from pytest import fixture, mark
from torch import Tensor, equal

from cellmorph.image import Dimension, pad, to_tensor
from cellmorph.training import prepare_target

@fixture(scope="function")
def img(width: int, height: int, rng: Generator) -> Image:
    arr = rng.integers(0, 255, (height, width, 4)).astype(uint8)
    return ImageFactory.fromarray(arr)


class TestPrepareTarget:
    """
    Test suite for :meth:`prepare_target`.
    """

    @mark.parametrize('width, height, max_size, padding, premultiply', [
        (128, 128, None, 0, False),
        (128, 128, None, 0, True),
        (256, 256, Dimension(40, 40), 16, False),
        (90, 90, Dimension(32, 32), 4, True)
    ])
    def test_prepare_target_with_no_max_size(
        self, img: Image, max_size: Dimension, padding: int,
        premultiply: bool
    ):
        from numpy import array
        eimg = ImageFactory.fromarray(array(img))

        if max_size:
            eimg.thumbnail(astuple(max_size), ImageFactory.LANCZOS)

        if padding:
            eimg = pad(eimg, padding)

        expected = to_tensor(eimg, premultiply=premultiply)
        result = prepare_target(img, max_size, padding, premultiply)

        assert result.shape == expected.shape

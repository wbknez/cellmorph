"""
Ensures that dimension initialization checks the parameters correctly.
"""
from numpy import float32, uint8
from numpy.random import Generator
from PIL import Image as ImageFactory
from PIL.Image import Image
from pytest import mark, raises
from torch import Tensor, from_numpy

from cellmorph.data import Dimension


class TestDimension:
    """
    Test suite for :class:`Dimension`.
    """

    @mark.parametrize('width', [ 0, -1 ])
    def test_dimension_raises_if_width_is_not_positive(self, width: int):
        with raises(ValueError):
            Dimension(width, 1)

    @mark.parametrize('height', [ 0, -1 ])
    def test_dimension_raises_if_height_is_not_positive(self, height: int):
        with raises(ValueError):
            Dimension(1, height)

    @mark.parametrize('width, height, color_channels', [
        (40, 40, 4),
        (32, 65, 4),
        (83, 19, 4)
    ])
    def test_dimension_created_successfully_from_an_image(self, img: Image):
        expected = Dimension(img.size[0], img.size[1])
        result = Dimension.from_image(img)

        assert result == expected

    @mark.parametrize('sample_count, state_channels, height, width', [
        (1, 16, 40, 40),
        (1, 8, 32, 65),
        (1, 97, 83, 19)
    ])
    def test_dimension_created_successfully_from_a_tensor(self, x: Tensor):
        expected = Dimension(x.shape[-1], x.shape[-2])
        result = Dimension.from_tensor(x)

        assert result == expected

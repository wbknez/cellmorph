"""
Ensures that transforming an image into either a larger or smaller version works
as intended.
"""
from dataclasses import astuple

from PIL import Image as ImageFactory
from PIL.Image import Image
from pytest import mark, raises
from torch import rand

from cellmorph.data import Dimension
from cellmorph.transforms.image import Resize


class TestResize:
    """
    Test suite for :class:`Resize`.
    """

    def test_constructor_raises_if_dimension_is_none(self):
        with raises(ValueError):
            Resize(None)

    def test_transform_raises_if_inpt_is_not_an_image(self):
        with raises(TypeError):
            Resize(Dimension(40, 40)).transform(rand((1, 16, 72, 72)))

    @mark.parametrize('width, height, color_channels, size', [
        (40, 40, 4, Dimension(40, 40)),
        (72, 17, 3, Dimension(120, 111)),
        (21, 29, 4, Dimension(11, 12))
    ])
    def test_transform_resizes_image(self, img: Image, size: Dimension):
        expected = img.resize(astuple(size), ImageFactory.LANCZOS)
        result = Resize(size).transform(img)

        assert result == expected

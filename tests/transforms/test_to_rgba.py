"""
Ensures that obtaining the RGBA channels of a tensor is correct.
"""
from PIL.Image import Image
from pytest import mark, raises
from torch import Tensor, equal, rand

from cellmorph.transforms.tensor import ToRgba


class TestToRgba:
    """"
    Test suite for :class:`ToRgba`.
    """

    @mark.parametrize('width, height, color_channels', [
        (72, 72, 4)
    ])
    def test_transform_raises_if_inpt_is_not_a_tensor(self, img: Image):
        with raises(TypeError):
            ToRgba().transform(img)

    def test_transform_raises_if_inpt_is_not_4D(self):
        with raises(ValueError):
            ToRgba().transform(rand((1, 1, 1, 1, 1)))

        with raises(ValueError):
            ToRgba().transform(rand((1, 1, 1)))

    @mark.parametrize('tnsr', [
        rand((1, 16, 72, 72)),
        rand((10, 6, 19, 27)),
    ])
    def test_transform_extracts_rgba_channels(self, tnsr: Tensor):
        expected = tnsr[:, :4]
        result = ToRgba().transform(tnsr)

        assert equal(result, expected)

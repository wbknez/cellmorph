"""
Ensures that obtaining the RGB channels of a tensor via alpha channel reduction
is correct.
"""
from PIL.Image import Image
from pytest import mark, raises
from torch import Tensor, equal, rand

from cellmorph.transforms.tensor import ToRgb


class TestToRgb:
    """"
    Test suite for :class:`ToRgb`.
    """

    @mark.parametrize('width, height, color_channels', [
        (72, 72, 4)
    ])
    def test_transform_raises_if_inpt_is_not_a_tensor(self, img: Image):
        with raises(TypeError):
            ToRgb().transform(img)

    def test_transform_raises_if_inpt_is_not_4D(self):
        with raises(ValueError):
            ToRgb().transform(rand((1, 1, 1, 1, 1)))

        with raises(ValueError):
            ToRgb().transform(rand((1, 1, 1)))

    @mark.parametrize('tnsr', [
        rand((1, 16, 72, 72)),
        rand((10, 6, 19, 27)),
    ])
    def test_transform_extracts_rgba_channels(self, tnsr: Tensor):
        expected = 1.0 - tnsr[:, 3:4] + tnsr[:, :3]
        result = ToRgb().transform(tnsr)

        assert equal(result, expected)

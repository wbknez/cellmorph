"""
Ensures that reducing the dimensionality of a tensor is correct.
"""
from PIL.Image import Image
from pytest import mark, raises
from torch import Tensor, equal, rand

from cellmorph.transforms.tensor import Squeeze


class TestSqueeze:
    """"
    Test suite for :class:`Squeeze`.
    """

    def test_constructor_raises_if_axis_is_negative(self):
        with raises(ValueError):
            Squeeze(-1)

    @mark.parametrize('width, height, color_channels', [
        (72, 72, 4)
    ])
    def test_transform_raises_if_inpt_is_not_a_tensor(self, img: Image):
        with raises(TypeError):
            Squeeze(0).transform(img)

    def test_transform_raises_if_axis_is_greater_than_inpt_shape(self):
        with raises(ValueError):
            Squeeze(5).transform(rand((3, 2, 4)))

    @mark.parametrize('axis, tnsr', [
        (0, rand((1, 16, 72, 72))),
        (3, rand((10, 16, 14, 19, 3)))
    ])
    def test_transform_reduce_dimension(self, axis: int, tnsr: Tensor):
        expected = tnsr.squeeze(axis)
        result = Squeeze(axis).transform(tnsr)

        assert equal(result, expected)

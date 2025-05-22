"""
Ensures that premultiplying RGB channels by their alpha works as expected.
"""
from PIL.Image import Image
from pytest import mark, raises
from torch import Tensor, equal, rand

from cellmorph.transforms.tensor import Premultiply


class TestPremultiply:
    """"
    Test suite for :class:`Premultiply`.
    """

    @mark.parametrize('width, height, color_channels', [
        (72, 72, 4)
    ])
    def test_transform_raises_if_inpt_is_not_a_tensor(self, img: Image):
        with raises(TypeError):
            Premultiply().transform(img)

    def test_transform_raises_if_inpt_is_not_3D_or_4D(self):
        with raises(ValueError):
            Premultiply().transform(rand((1, 1, 1, 1, 1)))

        with raises(ValueError):
            Premultiply().transform(rand((1, 1)))

    def test_transform_raises_if_number_of_channels_is_less_than_four(self):
        with raises(ValueError):
            Premultiply().transform(rand((1, 3, 16, 16)))

    @mark.parametrize('tnsr', [
        rand((1, 16, 72, 72)),
        rand((15, 47, 47))
    ])
    def test_transform_multiplies_rbg_by_alpha(self, tnsr: Tensor):

        expected = tnsr.clone()
        if len(tnsr.shape) == 4:
            expected[:, :3, ...] *= expected[:, 3:4, ...]
        else:
            expected[:3, ...] *= expected[3:4, ...]

        result = Premultiply().transform(tnsr)

        assert equal(result, expected)
        assert result.shape == tnsr.shape

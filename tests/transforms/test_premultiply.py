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
        rand((1, 16, 72, 72))
    ])
    def test_transform_multiplies_rbg_by_alpha(self, tnsr: Tensor):
        rgb = tnsr[:, :3]
        alpha = tnsr[:, 3:4]

        expected = rgb * alpha
        result = Premultiply().transform(tnsr)

        assert equal(result, expected)

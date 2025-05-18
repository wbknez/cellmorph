"""
Ensures that transforming a tensor to an image works as expected.
"""
from numpy import uint8
from PIL import Image as ImageFactory
from PIL.Image import Image
from pytest import mark, raises
from torch import Tensor, rand

from cellmorph.transforms.tensor import ToImage


class TestToTensor:
    """
    Test suite for :class:`ToTensor`.
    """

    @mark.parametrize('width, height, color_channels', [
        (72, 72, 4)
    ])
    def test_transform_raises_if_inpt_is_not_a_tensor(self, img: Image):
        with raises(TypeError):
            ToImage().transform(img)

    @mark.parametrize('tnsr', [
        rand((1, 4, 72, 72))
    ])
    def test_transform_converts_tensor_to_image(self, tnsr: Tensor):
        expected = tnsr.squeeze(0)
        expected = (expected * 255).permute(1, 2, 0).numpy()
        expected = ImageFactory.fromarray(expected.astype(uint8))

        result = ToImage().transform(tnsr.squeeze(0))

        assert result == expected

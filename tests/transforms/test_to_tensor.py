"""
Ensures that transforming an image to a tensor works as expected.
"""
from numpy import array, float32
from PIL.Image import Image
from pytest import mark, raises
from torch import equal, from_numpy, rand

from cellmorph.transforms.image import ToTensor


class TestToTensor:
    """
    Test suite for :class:`ToTensor`.
    """

    def test_transform_raises_if_inpt_is_not_an_image(self):
        with raises(TypeError):
            ToTensor().transform(rand((1, 16, 72, 72)))

    @mark.parametrize('width, height, color_channels', [
        (40, 40, 4),
        (72, 17, 3),
        (21, 29, 4)
    ])
    def test_transform_converts_image_to_tensor(self, img: Image):
        expected = from_numpy(array(img) / 255.0)
        expected = expected.permute(2, 0, 1).float()

        result = ToTensor().transform(img)

        assert equal(result, expected)

"""
Ensures that a pass-through transform does not perform any modifications.
"""
from PIL import Image as ImageFactory
from PIL.Image import Image
from pytest import mark, raises
from torch import equal, rand

from cellmorph.transforms import Pass


class TestPass:
    """
    Test suite for :class:`Pass`.
    """

    def test_transform_raises_if_inpt_is_not_an_image_or_tensor(self):
        with raises(TypeError):
            Pass().transform(42)

    @mark.parametrize("width, height, color_channels", [
        (40, 40, 4),
        (120, 97, 4),
        (19, 108, 4)
    ])
    def test_transform_returns_image_unmodified(self, img: Image):
        expected = img.copy()
        result = Pass().transform(img)

        assert result == expected
        assert not result is expected

    @mark.parametrize('shape', [
        (1, 16, 72, 72),
        (8, 23, 92, 92)
    ])
    def test_transform_returns_tensor_unmodified(self, shape: tuple[int]):
        inpt = rand(shape)

        expected = inpt.clone()
        result = Pass().transform(inpt)

        assert equal(expected, result)

"""
Ensures that transforming an image by padding each side works correctly.
"""
from PIL import Image as ImageFactory
from PIL.Image import Image
from pytest import mark, raises
from torch import rand

from cellmorph.transforms.image import Pad


class TestPad:
    """
    Test suite for :class:`Pad`.
    """

    def test_constructor_raises_if_amount_is_negative(self):
        with raises(ValueError):
            Pad(-1)

    def test_transform_raises_if_inpt_is_not_an_image(self):
        with raises(TypeError):
            Pad(16).transform(rand((1, 16, 72, 72)))

    @mark.parametrize("width, height, color_channels", [
        (40, 40, 4),
        (120, 97, 4),
        (19, 108, 4)
    ])
    def test_transform_returns_image_unmodified_if_padding_is_zero(
        self, img: Image
    ):
        expected = img.copy()
        result = Pad(0).transform(img)

        assert result == expected
        assert not result is expected

    @mark.parametrize("width, height, color_channels, padding", [
        (40, 40, 4, 16),
        (120, 97, 4, 9),
        (19, 108, 4, 88)
    ])
    def test_transform_returns_modified_image(self, padding: int, img: Image):
        width, height = img.size
        new_width, new_height = width + padding * 2, height + padding * 2

        expected = ImageFactory.new(img.mode, (new_width, new_height))
        expected.paste(img, (padding, padding))
        result = Pad(padding).transform(img)

        assert result == expected

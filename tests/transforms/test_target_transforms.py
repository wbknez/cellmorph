"""
Ensures that composed target transformations work as expected.
"""
from PIL.Image import Image
from pytest import mark
from torch import equal
from torchvision.transforms.v2 import Compose

from cellmorph.data import Dimension
from cellmorph.transforms import Pad, Premultiply, Resize, ToTensor


class TestTargetTransforms:
    """
    Test suite for multiple target-specific :class:`Transform` objects.
    """

    @mark.parametrize('width, height, color_channels', [
        (72, 72, 4)
    ])
    def test_target_transforms_compute_correctly(self, img: Image):
        transform = Compose([
            Resize(Dimension(40, 40)),
            Pad(16),
            ToTensor(),
            Premultiply()
        ])

        expected = Resize(Dimension(40, 40)).transform(img)
        expected = Pad(16).transform(expected)
        expected = ToTensor().transform(expected)
        expected = Premultiply().transform(expected)

        result = transform(img)

        assert equal(result, expected)

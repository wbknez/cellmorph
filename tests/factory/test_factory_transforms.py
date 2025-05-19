"""
Ensures that creating composed transformations from values in a configuration
file works correctly.
"""
from pytest import mark
from torch import equal
from torchvision.transforms.v2 import Compose

from cellmorph.config import Configuration
from cellmorph.factory import ConfigurationFactory
from cellmorph.transforms import Pad, Pass, Premultiply, Resize, ToTensor


class TestFactoryTransforms:
    """
    Test suite for :meth:`ConfigurationFactory.transforms`.
    """

    @mark.parametrize('values', [
        {
            "data": {
                "max_size": {
                    "width": 40,
                    "height": 40
                },
                "padding": 16
            },
            "premultiply": True
        },
        {
            "data": {
                "max_size": {
                    "width": 190,
                    "height": 42
                },
                "padding": 32
            },
            "premultiply": False
        },
    ])
    def test_transforms_creates_correct_transformations(
        self, config: Configuration
    ):
        result = ConfigurationFactory.transforms(config)

        assert isinstance(result.transforms[0], Resize)
        assert result.transforms[0]._size == config.data.max_size

        assert isinstance(result.transforms[1], Pad)
        assert result.transforms[1]._amount == config.data.padding

        assert isinstance(result.transforms[2], ToTensor)

        if config.data.premultiply:
            assert isinstance(result.transforms[3], Premultiply)
        else:
            assert isinstance(result.transforms[3], Pass)

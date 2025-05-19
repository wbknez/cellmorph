"""
Ensures that loading targets from images and emojis from values in a
configuration file works as expected.
"""
from pathlib import Path

from PIL import Image as ImageFactory
from PIL.Image import Image
from pytest import mark
from torch import equal
from torchvision.transforms.v2 import Compose

from cellmorph.config import YAML, Configuration
from cellmorph.data import Dimension
from cellmorph.emoji import CommonEmojis
from cellmorph.factory import ConfigurationFactory
from cellmorph.transforms import Pad, Pass, Premultiply, Resize, ToTensor


class TestFactoryTargets:
    """
    Test suite for :meth:`ConfigurationFactory.targets`.
    """

    @mark.parametrize('sample_count, width, height, color_channels, padding, premultiply', [
        (1024, 40, 40, 4, 16, True),
        (512, 62, 49, 4, 50, False)
    ])
    def test_targets_loads_correct_image_from_path(
        self, data: YAML, img: Image, tmpdir: Path, sample_count: int,
        width: int, height: int, padding: int, premultiply: bool
    ):
        img_path = tmpdir / "image.png"
        img.save(img_path)

        data["data"]["sample_count"] = sample_count
        data["data"]["target"] = str(img_path)
        data["data"]["max_size"] = {
            "width": width, "height": height
        }
        data["data"]["padding"] = padding
        data["data"]["premultiply"] = premultiply

        config = Configuration.from_dict(data)

        expected = Resize(config.data.max_size).transform(img)
        expected = Pad(config.data.padding).transform(img)
        expected = ToTensor().transform(expected)

        if config.data.premultiply:
            expected = Premultiply().transform(expected)

        result = ConfigurationFactory.targets(config)

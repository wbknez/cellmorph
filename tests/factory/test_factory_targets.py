"""
Ensures that loading targets from images and emojis works as expected.
"""
from pathlib import Path

from PIL import Image as ImageFactory
from PIL.Image import Image
from pytest import mark
from torch import equal

from cellmorph.config import YAML, Configuration
from cellmorph.emoji import CommonEmojis
from cellmorph.factory import ConfigurationFactory
from cellmorph.image import pad, to_tensor


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

        img.thumbnail((width, height), ImageFactory.LANCZOS)
        img = pad(img, padding)

        expected = to_tensor(img, premultiply=premultiply).repeat(
            (sample_count, 1, 1, 1)
        )
        result = ConfigurationFactory.targets(config)

        assert equal(result, expected)

        img.close()

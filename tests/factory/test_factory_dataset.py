"""
Ensures that creating an indexing dataset from values in a configuration file
works as expected.
"""
from dataclasses import astuple

from pytest import mark
from torch import equal, rand

from cellmorph.config import Configuration
from cellmorph.data import IndexingDataset
from cellmorph.factory import ConfigurationFactory


class TestFactoryDataset:
    """
    Test suite for :meth:`ConfigurationFactory.dataset`.
    """

    @mark.parametrize('values', [
        {
            "data": {
                "sample_count": 1024,
                "max_size": {
                    "width": 40,
                    "height": 40
                },
                "padding": 16
            },
            "model": {
                "state_channels": 16
            }
        },
        {
            "data": {
                "sample_count": 537,
                "max_size": {
                    "width": 23,
                    "height": 99
                },
                "padding": 33
            },
            "model": {
                "state_channels": 32
            }
        }
    ])
    def test_dataset_creates_correct_samples(
        self, config: Configuration
    ):
        width, height = astuple(config.data.max_size)

        width += config.data.padding * 2
        height += config.data.padding * 2

        targets = rand((config.data.sample_count, 4, height, width)).float()
        
        expected = IndexingDataset(config.data.sample_count,
                                   config.model.state_channels, targets)
        result = ConfigurationFactory.dataset(config, targets)

        assert equal(result.samples, expected.samples)
        assert equal(result.targets, expected.targets)

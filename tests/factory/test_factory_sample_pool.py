"""
Ensures that creating a sample pool from values in a configuration file works as
expected.
"""
from dataclasses import astuple

from pytest import mark
from torch import equal, rand

from cellmorph.config import Configuration
from cellmorph.data import IndexingDataset, SamplePool
from cellmorph.factory import ConfigurationFactory


class TestFactorySamplePool:
    """
    Test suite for :meth:`ConfigurationFactory.sample_pool`.
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
            },
            "train": {
                "batch_size": 8
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
            },
            "train": {
                "batch_size": 32,
            }
        }
    ])
    def test_sample_pool_creates_correct_dataset(
        self, config: Configuration
    ):
        width, height = astuple(config.data.max_size)

        width += config.data.padding * 2
        height += config.data.padding * 2

        targets = rand((config.data.sample_count, 4, height, width)).float()
        ds = IndexingDataset(config.data.sample_count,
                             config.model.state_channels, targets)
        
        expected = SamplePool(ds, config.train.batch_size)
        result = ConfigurationFactory.sample_pool(config, ds)

        assert result.batch_size == expected.batch_size
        assert equal(result.dataset.samples, expected.dataset.samples)
        assert equal(result.dataset.targets, expected.dataset.targets)

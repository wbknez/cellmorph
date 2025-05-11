"""
Ensures that creating an initial state from values in a configuration file works
as expected.
"""
from dataclasses import astuple

from pytest import mark
from torch import equal, zeros

from cellmorph.config import Configuration
from cellmorph.factory import ConfigurationFactory


class TestInitialState:
    """
    Test suite for :meth:`ConfigurationFactory.initial_state`.
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
            "model": {
                "state_channels": 16
            },
        },
        {
            "data": {
                "max_size": {
                    "width": 93,
                    "height": 17
                },
                "padding": 44
            },
            "model": {
                "state_channels": 32
            },
        }
    ])
    def test_initial_state_creates_correct_empty_seed(
        self, config: Configuration
    ):
        state_channels = config.model.state_channels
        width, height = astuple(config.data.max_size)
        padding = config.data.padding

        width = width + 2 * padding
        height = height + 2 * padding

        expected = zeros((1, state_channels, height, width)).float()

        expected[:, 3:, height // 2, width // 2] = 1.0

        result = ConfigurationFactory.initial_state(config)

        assert result.shape == expected.shape
        assert equal(result, expected)

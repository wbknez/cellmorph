"""
Ensures that creating an update strategy from values in a configuration file
works as expected.
"""
from pytest import mark
from torch import equal

from cellmorph.config import Configuration
from cellmorph.data import (
    GrowthStrategy,
    PersistentStrategy,
    RegenerativeStrategy
)
from cellmorph.factory import ConfigurationFactory


class TestFactoryStrategy:
    """
    Test suite for :meth:`ConfigurationFactory.strategy`.
    """

    @mark.parametrize('values', [
        {
            "data": {
                "max_size": {
                    "width": 40,
                    "height": 40
                },
            },
            "model": {
                "state_channels": 16
            },
            "train": {
                "strategy": "growth"
            }
        },
        {
            "data": {
                "max_size": {
                    "width": 32,
                    "height": 64
                },
            },
            "model": {
                "state_channels": 99
            },
            "train": {
                "strategy": "growth"
            }
        }
    ])
    def test_strategize_creates_correct_growth_strategy(
        self, config: Configuration
    ):
        initial_state = ConfigurationFactory.initial_state(config)

        expected = GrowthStrategy(initial_state)
        result = ConfigurationFactory.strategy(config)

        assert isinstance(result, GrowthStrategy)
        assert equal(result._initial_state, expected._initial_state)

    @mark.parametrize('values', [
        {
            "data": {
                "max_size": {
                    "width": 40,
                    "height": 40
                },
            },
            "model": {
                "state_channels": 16
            },
            "train": {
                "strategy": "persistent",
                "reset_count": 1
            }
        },
        {
            "data": {
                "max_size": {
                    "width": 32,
                    "height": 64
                },
            },
            "model": {
                "state_channels": 99
            },
            "train": {
                "strategy": "persistent",
                "reset_count": 12
            }
        }
    ])
    def test_strategize_creates_correct_persistent_strategy(
        self, config: Configuration
    ):
        initial_state = ConfigurationFactory.initial_state(config)
        reset_count = config.train.reset_count

        expected = PersistentStrategy(initial_state, reset_count)
        result = ConfigurationFactory.strategy(config)

        assert isinstance(result, PersistentStrategy)
        assert result._reset_count == expected._reset_count
        assert equal(result._initial_state, expected._initial_state)

    @mark.parametrize('values', [
        {
            "data": {
                "max_size": {
                    "width": 40,
                    "height": 40
                },
            },
            "model": {
                "state_channels": 16
            },
            "train": {
                "strategy": "regenerative",
                "reset_count": 1,
                "damage_count": 3
            }
        },
        {
            "data": {
                "max_size": {
                    "width": 32,
                    "height": 64
                },
            },
            "model": {
                "state_channels": 99
            },
            "train": {
                "strategy": "regenerative",
                "reset_count": 12,
                "damage_count": 32
            }
        }
    ])
    def test_strategize_creates_correct_regenerative_strategy(
        self, config: Configuration
    ):
        initial_state = ConfigurationFactory.initial_state(config)
        reset_count = config.train.reset_count
        damage_count = config.train.damage_count

        expected = RegenerativeStrategy(initial_state, reset_count,
                                        damage_count)
        result = ConfigurationFactory.strategy(config)

        assert isinstance(result, RegenerativeStrategy)
        assert result._reset_count == expected._reset_count
        assert equal(result._initial_state, expected._initial_state)

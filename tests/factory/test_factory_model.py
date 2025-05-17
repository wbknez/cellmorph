"""
Ensures that creating a model from values in a configuration file works as
expected.
"""
from dataclasses import astuple
from pathlib import Path

from pytest import mark
from torch import equal, manual_seed, rand_like

from cellmorph.config import Configuration
from cellmorph.factory import ConfigurationFactory
from cellmorph.model import Model
from cellmorph.utils import random_bytes


class TestFactoryModel:
    """
    Test suite for :meth:`ConfigurationFactory.model`.
    """

    @mark.parametrize('values', [
        {
            "model": {
                "state_channels": 16,
                "hidden_channels": 128,
                "update_rate": 0.5,
                "step_size": 0.5,
                "rotation": 0.0,
                "threshold": 0.1,
                "normalize_kernel": False,
                "use_bias": False
            }
        },
        {
            "model": {
                "state_channels": 28,
                "hidden_channels": 208,
                "update_rate": 0.35,
                "step_size": 0.63,
                "rotation": 3.1417,
                "threshold": 0.3,
                "normalize_kernel": True,
                "use_bias": True
            }
        }
    ])
    def test_model_creates_correct_model(
        self, config: Configuration
    ):
        seed = random_bytes(8)

        manual_seed(seed)
        expected = Model(
            state_channels=config.model.state_channels,
            hidden_channels=config.model.hidden_channels,
            update_rate=config.model.update_rate,
            step_size=config.model.step_size,
            rotation=config.model.rotation,
            threshold=config.model.threshold,
            normalize_kernel=config.model.normalize_kernel,
            use_bias=config.model.use_bias
        )

        manual_seed(seed)
        result = ConfigurationFactory.model(config)

        assert equal(result.perception_rule._weights.data,
                     expected.perception_rule._weights.data)
        for p0, p1 in zip(expected.parameters(), result.parameters()):
            assert equal(p0, p1)

    @mark.parametrize('values', [
        {
            "model": {
                "state_channels": 16,
                "hidden_channels": 128,
                "update_rate": 0.5,
                "step_size": 0.5,
                "rotation": 0.0,
                "threshold": 0.1,
                "normalize_kernel": False,
                "use_bias": False
            }
        },
        {
            "model": {
                "state_channels": 28,
                "hidden_channels": 208,
                "update_rate": 0.35,
                "step_size": 0.63,
                "rotation": 3.1417,
                "threshold": 0.3,
                "normalize_kernel": True,
                "use_bias": True
            }
        }
    ])
    def test_model_creates_correct_model_and_loads_correct_weights(
        self, config: Configuration, tmpdir: Path
    ):
        weights_path = tmpdir / "weights.pth"
        model = Model(
            state_channels=config.model.state_channels,
            hidden_channels=config.model.hidden_channels,
            update_rate=config.model.update_rate,
            step_size=config.model.step_size,
            rotation=config.model.rotation,
            threshold=config.model.threshold,
            normalize_kernel=config.model.normalize_kernel,
            use_bias=config.model.use_bias
        )

        model._updater._layers[0].weight.data = rand_like(
            model._updater._layers[0].weight.data
        )
        model._updater._layers[2].weight.data = rand_like(
            model._updater._layers[2].weight.data
        )

        model.save(weights_path)

        expected = model
        result = ConfigurationFactory.model(config, weights_path)

        assert equal(result.perception_rule._weights.data,
                     expected.perception_rule._weights.data)
        for p0, p1 in zip(expected.parameters(), result.parameters()):
            assert equal(p0, p1)

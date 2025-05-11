"""
Ensures that creating a trainer object from a configuration file works as
expected.
"""
from pytest import mark
from torch import manual_seed

from cellmorph.config import Configuration
from cellmorph.factory import ConfigurationFactory
from cellmorph.training import Trainer
from cellmorph.utils import random_bytes


class TestFactoryTrainer:
    """
    Test suite for :meth:`ConfigurationFactory.trainer`.
    """

    @mark.parametrize('values', [
        {
            "model": {
                "state_channels": 16,
                "hidden_channels": 128,
                "normalize_kernel": False,
                "padding": 0,
                "rotation": 0.0,
                "step_size": 1.0,
                "threshold": 0.1,
                "update_rate": 0.5,
                "use_bias": False
            },
            "optim": {
                "milestones": [ 2000 ],
                "gamma": 0.1,
                "learning_rate": 0.002,
                "gradient_cutoff": 20
            },
            "train": {
                "strategy": "persistent",
                "batch_size": 8,
                "steps": { "min": 64, "max": 96 },
                "epochs": 8000
            }
        }
    ])
    def test_trainer_creates_correct_trainer(self, config: Configuration):
        seed = random_bytes(8)

        model = ConfigurationFactory.model(config)
        strategy = ConfigurationFactory.strategy(config)

        manual_seed(seed)
        expected = Trainer(
            model,
            strategy,
            config.train.steps,
            config.optim.learning_rate,
            config.optim.milestones,
            config.optim.gamma,
            config.optim.gradient_cutoff
        )
        
        manual_seed(seed)
        result = ConfigurationFactory.trainer(config, model)

        assert isinstance(result._strategy, type(expected._strategy))
        assert result._steps == expected._steps
        assert result._gradient_cutoff == expected._gradient_cutoff
        assert result._optimizer.param_groups == \
            expected._optimizer.param_groups
        assert result._scheduler.milestones == expected._scheduler.milestones
        assert result._scheduler.gamma == expected._scheduler.gamma


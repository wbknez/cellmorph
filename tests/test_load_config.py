"""

"""
from pathlib import Path
from typing import Any

from numpy import pi
from numpy.random import Generator
from pytest import fixture
from yaml import dump

from cellmorph.config import (
    Configuration,
    DataConfiguration,
    ModelConfiguration,
    OptimizerConfiguration,
    TrainingConfiguration,
    load_config
)
from cellmorph.image import Dimension
from cellmorph.training import Interval


@fixture(scope="function")
def data(rng: Generator) -> dict[str, Any]:
    return {
        "name": "some_model_name",
        "test_frequency": int(rng.integers(0, 200)),
        "output_dir": None,
        "data": {
            "max_size": (int(rng.integers(32, 64)),
                         int(rng.integers(32, 64))),
            "padding": int(rng.integers(0, 32)),
            "premultiply": bool(rng.binomial(1, 0.5)),
            "cache_dir": None
        },
        "model": {
            "state_channels": int(rng.integers(5, 32)),
            "intermediate_channels": int(rng.integers(32, 128)),
            "normalize_kernel": bool(rng.binomial(1, 0.5)),
            "padding": int(rng.integers(0, 16)),
            "rotation": float(rng.random() * 2 * pi),
            "step_size": float(rng.random() * 2),
            "threshold": float(rng.random()),
            "update_rate": float(rng.random()),
            "use_bias": bool(rng.binomial(1, 0.5))
        },
        "optim": {
            "milestones": [ int(rng.integers(1, 10)),
                            int(rng.integers(100, 200)) ],
            "gamma": float(rng.random()),
            "learning_rate": float(rng.random()),
            "gradient_cutoff": int(rng.integers(1, 20))
        },
        "train": {
            "batch_size": int(rng.integers(1, 32)),
            "interval": ( int(rng.integers(1, 32)), int(rng.integers(33, 64))),
            "epochs": int(rng.integers(2000, 8000))
        }
    }


class TestLoadConfig:
    """
    Test suite for :meth:`load_config`.
    """

    def test_load_config_returns_valid_data(
        self, data: dict[str, Any], tmpdir: Path
    ):
        config_path = tmpdir / "config.yml"

        with open(config_path, "w") as f:
            f.write(dump(data))

        expected = Configuration(
            name=data["name"],
            test_frequency=data["test_frequency"],
            output_dir=None,
            data=DataConfiguration(
                max_size=Dimension(*data["data"]["max_size"]),
                padding=data["data"]["padding"],
                premultiply=data["data"]["premultiply"],
                cache_dir=None
            ),
            model=ModelConfiguration(
                state_channels=data["model"]["state_channels"],
                intermediate_channels=data["model"]["intermediate_channels"],
                normalize_kernel=data["model"]["normalize_kernel"],
                padding=data["model"]["padding"],
                rotation=data["model"]["rotation"],
                step_size=data["model"]["step_size"],
                threshold=data["model"]["threshold"],
                update_rate=data["model"]["update_rate"],
                use_bias=data["model"]["use_bias"]
            ),
            optim=OptimizerConfiguration(
                milestones=data["optim"]["milestones"],
                gamma=data["optim"]["gamma"],
                learning_rate=data["optim"]["learning_rate"],
                gradient_cutoff=data["optim"]["gradient_cutoff"]
            ),
            train=TrainingConfiguration(
                batch_size=data["train"]["batch_size"],
                interval=Interval(*data["train"]["interval"]),
                epochs=data["train"]["epochs"]
            )
        )
        result = load_config(config_path)

        assert result == expected

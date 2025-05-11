"""

"""
from pathlib import Path

from numpy import pi
from numpy.random import Generator
from pytest import fixture

from cellmorph.config import YAML, Configuration
from cellmorph.utils import combine_dicts


@fixture(scope="function")
def cache_dir(random_path: Path) -> Path:
    return random_path


@fixture(scope="function")
def output_dir(random_path: Path) -> Path:
    return random_path


@fixture(scope="function")
def data(output_dir: Path, cache_dir: Path, rng: Generator) -> YAML:
    return {
        "name": "some_model_name",
        "test_frequency": int(rng.integers(0, 200)),
        "output_dir": output_dir,
        "data": {
            "target": "some_target",
            "sample_count": int(rng.integers(8, 32)),
            "max_size": { "width": int(rng.integers(32, 64)),
                          "height": int(rng.integers(32, 64)) },
            "padding": int(rng.integers(0, 32)),
            "premultiply": bool(rng.binomial(1, 0.5)),
            "cache_dir": cache_dir
        },
        "model": {
            "state_channels": int(rng.integers(5, 32)),
            "hidden_channels": int(rng.integers(32, 128)),
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
            "dataset": "persistent",
            "batch_size": int(rng.integers(1, 32)),
            "steps": { "min": int(rng.integers(1, 32)),
                       "max": int(rng.integers(33, 64)) },
            "epochs": int(rng.integers(2000, 8000))
        }
    }


@fixture(scope="function")
def config(values: YAML, data: YAML) -> Configuration:
    """

    """
    return Configuration.from_dict(combine_dicts(data, values))

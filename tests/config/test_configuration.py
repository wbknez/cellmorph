"""
Ensures that loading configuration(s) from file(s) works correctly.
"""
from pathlib import Path
from string import ascii_letters, digits

from numpy import arange, pi
from numpy.random import Generator
from pytest import fixture
from yaml import dump

from cellmorph.config import (
    Configuration,
    DataConfiguration,
    Interval,
    ModelConfiguration,
    OptimizerConfiguration,
    TrainingConfiguration,
    YAML
)
from cellmorph.emoji import CommonEmojis
from cellmorph.image import Dimension


@fixture(scope="function")
def emoji(rng: Generator) -> str:
    choices = list(CommonEmojis)
    indices = arange(len(choices))

    return choices[rng.choice(indices)]


@fixture(scope="function")
def cache_dir(random_path: Path) -> Path:
    return random_path


@fixture(scope="function")
def output_dir(random_path: Path) -> Path:
    return random_path


@fixture(scope="function")
def data(emoji: str, output_dir: Path, cache_dir: Path,
         rng: Generator) -> YAML:
    return {
        "name": "some_model_name",
        "test_frequency": int(rng.integers(0, 200)),
        "output_dir": output_dir,
        "data": {
            "target": emoji,
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
            "strategy": "persistent",
            "batch_size": int(rng.integers(1, 32)),
            "steps": { "min": int(rng.integers(1, 32)),
                       "max": int(rng.integers(33, 64)) },
            "epochs": int(rng.integers(2000, 8000)),
            "reset_count": int(rng.integers(0, 5)),
            "damage_count": int(rng.integers(0, 10))
        }
    }


@fixture(scope="function")
def config(data: YAML) -> Configuration:
    max_size = data["data"]["max_size"]
    steps = data["train"]["steps"]

    print(f"Max_size: { max_size }")
    print(f"Steps: { steps }")

    return Configuration(
        name=data["name"],
        test_frequency=data["test_frequency"],
        output_dir=data["output_dir"],
        data=DataConfiguration(
            target=data["data"]["target"],
            sample_count=data["data"]["sample_count"],
            max_size=Dimension(max_size["width"], max_size["height"]),
            padding=data["data"]["padding"],
            premultiply=data["data"]["premultiply"],
            cache_dir=data["data"]["cache_dir"]
        ),
        model=ModelConfiguration(
            state_channels=data["model"]["state_channels"],
            hidden_channels=data["model"]["hidden_channels"],
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
            strategy=data["train"]["strategy"],
            batch_size=data["train"]["batch_size"],
            steps=Interval(steps["min"], steps["max"]),
            epochs=data["train"]["epochs"],
            reset_count=data["train"]["reset_count"],
            damage_count=data["train"]["damage_count"]
        )
    )


class TestConfiguration:
    """
    Test suite for :class:`Configuration`.
    """

    def test_configuration_saves_to_file_correctly(
        self, config: Configuration, tmp_path: Path
    ):
        config_path = tmp_path / "config.yml"
        config.save(config_path)

        result = Configuration.from_file(config_path)

        assert result == config

    def test_configuration_from_dict_returns_valid_data(
        self, data: YAML
    ):
        expected = Configuration(
            name=data["name"],
            test_frequency=data["test_frequency"],
            output_dir=data["output_dir"],
            data=DataConfiguration(
                target=data["data"]["target"],
                sample_count=data["data"]["sample_count"],
                max_size=Dimension(
                    width=data["data"]["max_size"]["width"],
                    height=data["data"]["max_size"]["height"]
                ),
                padding=data["data"]["padding"],
                premultiply=data["data"]["premultiply"],
                cache_dir=data["data"]["cache_dir"]
            ),
            model=ModelConfiguration(
                state_channels=data["model"]["state_channels"],
                hidden_channels=data["model"]["hidden_channels"],
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
                strategy=data["train"]["strategy"],
                batch_size=data["train"]["batch_size"],
                steps=Interval(
                    a=data["train"]["steps"]["min"],
                    b=data["train"]["steps"]["max"],
                ),
                epochs=data["train"]["epochs"],
                reset_count=data["train"]["reset_count"],
                damage_count=data["train"]["damage_count"]
            )
        )
        result = Configuration.from_dict(data)

        assert result == expected

    def test_configuration_from_file_returns_valid_data(
        self, data: YAML, tmpdir: Path
    ):
        config_path = tmpdir / "config.yml"

        with open(config_path, "w") as f:
            f.write(dump(data))

        expected = Configuration.from_dict(data)
        result = Configuration.from_file(config_path)

        assert result == expected

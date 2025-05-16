"""
A collection of classes and functions to load model configuration data from YAML
file(s).
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path

from numpy import pi
from yaml import CLoader, dump, load

from cellmorph.emoji import CommonEmojis
from cellmorph.image import Dimension
from cellmorph.training import Interval


type Value = float | int | str | Path
"""All potential value types expected from a YAML file."""


type YAML = dict[str, Value | list[Value] | YAML]
"""A collection of configuration values obtained from a YAML file."""


def get_bool(yaml: YAML, name: str, default_value: bool) -> bool:
    """
    Converts a single configuration value into a boolean.

    Args:
        yaml: The data values to convert.
        name: The name of the configuration value.
        default_value: The value to use if no value can be found.

    Returns:
        A configuration value as a boolean.
    """
    value = yaml.get(name, default_value)

    if not isinstance(value, bool | int | str):
        raise ValueError(f"Value is not a boolean or string: {value}.")

    try:
        if isinstance(value, str):
            if value.lower() in [ "true", "1" ]:
                value = True
            elif value.lower() in [ "false", "0" ]:
                value = False
            else:
                raise TypeError()

        return bool(value)
    except (TypeError, ValueError):
        raise ValueError(f"Cannot convert value to a boolean: {value}.")


def get_dimension(yaml: YAML, name: str, default_value: Dimension) -> Dimension:
    """
    Converts a single configuration value into a :class:`Dimension`.

    Args:
        yaml: The data values to convert.
        name: The name of the configuration value.
        default_value: The value to use if no value can be found.

    Returns:
        A configuration value as a :class:`Dimension`.
    """
    if not name in yaml:
        return default_value

    try:
        width = get_int(yaml[name], "width", default_value.width)
        height = get_int(yaml[name], "height", default_value.height)

        return Dimension(width, height)
    except AttributeError:
        raise ValueError(f"Missing one or more attributes for value: {name}.")


def get_float(yaml: YAML, name: str, default_value: float) -> float:
    """
    Converts a single configuration value into a float.

    Args:
        yaml: The data values to convert.
        name: The name of the configuration value.
        default_value: The value to use if no value can be found.

    Returns:
        A configuration value as a float.
    """
    value = yaml.get(name, default_value)

    if not isinstance(value, float | int | str):
        raise ValueError(f"Value is not a float, integer, or string: {value}.")

    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Cannot convert value to a float: {value}.")


def get_int(yaml: YAML, name: str, default_value: int) -> int:
    """
    Converts a single configuration value into an integer.

    Args:
        yaml: The data values to convert.
        name: The name of the configuration value.
        default_value: The value to use if no value can be found.

    Returns:
        A configuration value as an integer.
    """
    value = yaml.get(name, default_value)

    if not isinstance(value, int | str):
        raise ValueError(f"Value is not an integer or string: {value}.")

    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Cannot convert value to an integer: {value}.")


def get_interval(yaml: YAML, name: str, default_value: Interval) -> Interval:
    """
    Converts a single configuration value into a :class:`Interval`.

    Args:
        yaml: The data values to convert.
        name: The name of the configuration value.
        default_value: The value to use if no value can be found.

    Returns:
        A configuration value as a :class:`Interval`.
    """
    if not name in yaml:
        return default_value

    try:
        a = get_int(yaml[name], "min", default_value.a)
        b = get_int(yaml[name], "max", default_value.b)

        return Interval(a, b)
    except AttributeError:
        raise ValueError(f"Missing one or more attributes for value: {name}.")


def get_list(yaml: YAML, name: str, ctype: Value,
             default_values: ValueList) -> ValueList:
    """
    Converts a single configuration value into a :class:`list` of a particular
    type.

    Args:
        yaml: The data values to convert.
        name: The name of the configuration value.
        default_value: The value to use if no value can be found.

    Returns:
        A configuration value as a typed :class:`list`.
    """
    if not name in yaml:
        return default_value

    values = yaml[name]

    if not isinstance(values, list):
        raise ValueError(f"Value is not a list: {value}.")

    for i in range(len(values)):
        try:
            values[i] = ctype(values[i])
        except (TypeError, ValueError):
            cname = ctype.__name__
            raise ValueError(f"Could not convert item at index {i} to {cname}. "
                             f"{values}.")

    return values


def get_path(yaml: YAML, name: str, default_value: Path) -> Path:
    """
    Converts a single configuration value into a :class:`Path`.

    Args:
        yaml: The data values to convert.
        name: The name of the configuration value.
        default_value: The value to use if no value can be found.

    Returns:
        A configuration value as a :class:`Path`.
    """
    value = yaml.get(name, default_value)

    if isinstance(value, str):
        value = Path(value)

    if not isinstance(value, Path):
        raise ValueError(f"Count not convert value to path: {value}.")

    return value


def get_str(yaml: YAML, name: str, default_value: str) -> str:
    """
    Converts a single configuration value into a string.

    Args:
        yaml: The data values to convert.
        name: The name of the configuration value.
        default_value: The value to use if no value can be found.

    Returns:
        A configuration value as a string.
    """
    value = yaml.get(name, default_value)

    if not isinstance(value, str):
        raise ValueError(f"Value is not a string: {value}.")

    return value


@dataclass(frozen=True, slots=True)
class DataConfiguration:
    """
    A collection of configuration options specific to modeling data.
    """

    target: str
    """
    The training target image.

    This can be one of three items:
        1. An emoji character, or
        2. An emoji hexadecimal string, or
        3. A full file path to a custom image.
    """

    sample_count: int
    """The total number of samples in the dataset."""

    max_size: Dimension
    """The maximum size of un-padded data."""

    padding: int
    """The amount of pixels to surround the data with from all sides."""

    premultiply: bool
    """
    Whether to premultiply the RGB components of the data by any alpha values.
    """

    cache_dir: Path | None
    """The directory to cache any downloaded images (specifically, emojis)."""

    def __post_init__(self):
        if not self.cache_dir:
            raise ValueError("No cache directory defined.")

        if self.cache_dir and isinstance(self.cache_dir, str):
            new_dir = Path(self.cache_dir).resolve()

            object.__setattr__(self, "cache_dir", new_dir)

        if self.sample_count < 1:
            raise ValueError(f"Sample count must be positive: "
                             f"{self.sample_count}.")

        if not self.target:
            raise ValueError("Target must be defined for training to occur.")


@dataclass(frozen=True, slots=True)
class ModelConfiguration:
    """
    A collection of configuration options specific to the CNA model itself.
    """

    state_channels: int
    """The state space size per automata."""

    hidden_channels: int
    """The size of the intermediate state space per automata (layer 1 -> 2)."""

    normalize_kernel: bool
    """Whether to normalize the Sobel kernel."""

    rotation: float
    """The target image's angle of rotation."""

    step_size: int
    """The relative time dialation per step."""

    threshold: float
    """The automata activity cutoff value."""

    update_rate: float
    """The update probability per automata."""

    use_bias: bool
    """Whether to track and use learned bias."""

    def __post_init__(self):
        """
        Ensure that all configuration values are within bounds.
        """
        if self.state_channels < 5:
            raise ValueError(f"State channels must be greater than four: "
                             f"{self.state_channels}.")

        if self.hidden_channels < 1:
            raise ValueError(f"Intermediate channels must be positive: "
                             f"{self.intermediate_channels}.")

        if not 0.0 <= self.rotation < (2.0 * pi):
            raise ValueError(f"Rotation must be in [0, 2pi]: {self.rotation}.")

        if self.step_size <= 0.0:
            raise ValueError(f"Step size must be positive: {self.step_size}.")

        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"Threshold must be in (0, 1]: {self.threshold}.")

        if not 0.0 < self.update_rate <= 1.0:
            raise ValueError(f"Update rate must be in (0, 1]: "
                             f"{self.update_rate}.")


@dataclass(frozen=True, slots=True)
class OptimizerConfiguration:
    """
    A collection of configuration options specific to model optimization.
    """

    milestones: list[int]
    """A list of epochs denoting when the learning rate should decay."""

    learning_rate: float
    """The initial learning rate (before decay)."""

    gamma: float
    """The learning rate decay rate."""

    gradient_cutoff: float
    """The maximum value of a gradient normal."""

    def __post_init__(self):
        """
        Ensure that all configuration values are within bounds.
        """
        if not self.milestones or (len(self.milestones) == 0):
            raise ValueError("At least one milestone must be defined.")

        if any([milestone < 1 for milestone in self.milestones]):
            raise ValueError(f"Every milestone must be positive: "
                             f"{self.milestones}.")

        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive: "
                             f"{self.learning_rate}.")

        if self.gamma <= 0:
            raise ValueError(f"Gamma must be positive: {self.gamma}.")

        if self.gradient_cutoff <= 0:
            raise ValueError(f"Gradient cuttoff must be positive: "
                             f"{self.gradient_cutoff}.")

@dataclass(frozen=True, slots=True)
class TrainingConfiguration:
    """
    A collection of configuration options specific to model training.
    """

    strategy: str
    """The type of update strategy to modify a training dataset with."""

    batch_size: int
    """The number of training samples per iteration."""

    steps: Interval
    """The minimum and maximum number of iterations per training target."""

    epochs: int
    """The maximum number of training iterations to complete"""

    reset_count: int
    """The number of worst samples to reinitialze after an epoch."""

    damage_count: int
    """The number of best samples to apply a damage mask to after an epoch."""

    def __post_init__(self):
        """
        Ensure that all configuration values are within bounds.
        """
        if not self.strategy:
            raise ValueError("Update strategy is not defined.")

        if self.batch_size < 1:
            raise ValueError(f"Batch size must be positive: {self.batch_size}.")

        if not self.steps:
            raise ValueError("Step interval is not defined.")

        if self.epochs < 1:
            raise ValueError(f"Epochs must be positive: {self.epochs}.")

        if self.reset_count < 0:
            raise ValueError(f"Reset count cannot be negative: "
                             f"{self.reset_count}.")

        if self.damage_count < 0:
            raise ValueError(f"Damage count cannot be negative: "
                             f"{self.damage_count}.")


@dataclass(frozen=True, slots=True)
class Configuration:
    """
    A collection of all possible model configuration values.
    """

    name: str
    """The name of the model; used for unique file naming purposes."""

    test_frequency: int
    """How often to create/use test cases on models in training."""

    output_dir: Path
    """The path to a directory in which to place any generated output."""

    data: DataConfiguration
    """Any data specific configuration."""

    model: ModelConfiguration
    """Any model specific configuration."""

    optim: OptimizerConfiguration
    """Any model optimization configuration."""

    train: TrainingConfiguration
    """Any training specific configuration."""

    def __post_init__(self):
        if not self.name:
            raise ValueError("Unique model name must be defined.")

        if self.test_frequency < 0:
            raise ValueError(f"Test frequency cannot be negative: "
                             f"{self.test_frequency}.")

        if not self.output_dir:
            raise ValueError("Output directory must be defined.")

        if self.output_dir and isinstance(self.output_dir, str):
            new_dir = Path(self.output_dir).resolve()

            object.__setattr__(self, "output_dir", new_dir)

    def save(self, file_path: Path):
        """
        Saves all values in this configuration to a YAML file.

        Args:
            file_path: The location to write to.
        """
        values = self.to_dict()

        values["output_dir"] = str(self.output_dir)
        values["data"]["cache_dir"] = str(self.data.cache_dir)

        with open(file_path, "w") as f:
            f.write(dump(values))

    def to_dict(self) -> YAML:
        """
        Converts this configuration to a dictionary of key/value pairs suitable
        for writing to a YAML file or other forms of modification.

        Returns:
            All configuration values as a dictionary.
        """
        values = asdict(self)

        values["data"]["max_size"] = {
            "width": self.data.max_size.width,
            "height": self.data.max_size.height
        }
        values["train"]["steps"] = {
            "min": self.train.steps.a,
            "max": self.train.steps.b
        }

        return values


    @classmethod
    def from_dict(cls, values: YAML) -> Configuration:
        """
        Creates a :class:`Configuration` from a dictionary containing parameter
        names and their associated values.

        Args:
            values: A mapping of configuration names to values.

        Returns:
            A new configuration object.
        """
        data = values["data"]
        model = values["model"]
        optim = values["optim"]
        train = values["train"]

        return Configuration(
            name=get_str(values, "name", "experiment"),
            test_frequency=get_int(values, "test_frequency", 0),
            output_dir=get_path(values, "output_dir", Path("./output")),
            data=DataConfiguration(
                target=get_str(data, "target", CommonEmojis.BUTTERFLY),
                sample_count=get_int(data, "sample_count", 1024),
                max_size=get_dimension(data, "max_size", Dimension(40, 40)),
                padding=get_int(data, "padding", 16),
                premultiply=get_bool(data, "premultiply", True),
                cache_dir=get_path(data, "cache_dir", Path("./emoji_cache/"))
            ),
            model=ModelConfiguration(
                state_channels=get_int(model, "state_channels", 16),
                hidden_channels=get_int(model, "hidden_channels", 128),
                normalize_kernel=get_bool(model, "normalize_kernel", False),
                rotation=get_float(model, "rotation", 0.0),
                step_size=get_float(model, "step_size", 1.0),
                threshold=get_float(model, "threshold", 0.1),
                update_rate=get_float(model, "update_rate", 0.5),
                use_bias=get_bool(model, "use_bias", False)
            ),
            optim=OptimizerConfiguration(
                milestones=get_list(optim, "milestones", int, [ 2000 ]),
                gamma=get_float(optim, "gamma", 0.1),
                learning_rate=get_float(optim, "learning_rate", 2e-3),
                gradient_cutoff=get_float(optim, "gradient_cutoff", 20.0)
            ),
            train=TrainingConfiguration(
                strategy=get_str(train, "strategy", "Persistent"),
                batch_size=get_int(train, "batch_size", 8),
                steps=get_interval(train, "steps", Interval(64, 96)),
                epochs=get_int(train, "epochs", 8000),
                reset_count=get_int(train, "reset_count", 1),
                damage_count=get_int(train, "damage_count", 3)
            )
        )

    @classmethod
    def from_file(cls, config_path: Path) -> Configuration:
        """
        Reads a YAML file filled with keys and values and converts it to a model
        configuration.

        Args:
            config_path: The path to the configuration file to load.

        Returns:
            A successfully loaded configuration.
        """
        with open(config_path, "r") as f:
            values = load(f, CLoader)
            return Configuration.from_dict(values)

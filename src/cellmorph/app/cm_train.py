#!/usr/bin/env python3


"""

Command-line Arguments:
    - "-c"/"--config-file": The path to a configuration file containing all
      necessary values and parameters to both create a model and train it.  This
      path may be either absolute or relative; however, it must obviously exist
      in order for the program to continue.
    - "-l"/"--logging-level": The global logging level that determines which
      logging messages appear in the model-specific log file.  Any option other
      than "off" will filter messages such that if a message's level is beneath
      the global level then it will be ignored.  Please note that using "off" to
      disable logging will result in silencing *all* logging output, including
      any exceptions that are thrown in functions decorated with
      `@logger.catch` with the sole exception of `main`.
    - "-q"/"--quiet": Silences all command-line textual output.  By default,
      this program produces a nicely formatted progress bar to visually track
      the training progress of a model.  Invoking this option removes this as
      well as any `stdout` or `stderr` specific output such as separate logging
      handlers that write to those streams.
"""
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from pathlib import Path
from sys import exit, stdout

from loguru import logger
from numpy import array, savetxt
from tqdm import tqdm

from cellmorph.config import Configuration
from cellmorph.factory import ConfigurationFactory
from cellmorph.model import Model
from cellmorph.utils import Stopwatch, TqdmSink, choose_device, unique_name


LOG_FORMAT = (
    "<green>{time:HH:mm:ss}</green> "
    "<level>{level: <8}</level> "
    "<<cyan>{function}</cyan>:<cyan>{line}</cyan>> "
    "<level>{message}</level>"
)
"""The logging message format for this project."""


@dataclass(slots=True)
class Losses:
    """
    A wrapper around a collection of generated per-epoch loss values.
    """

    _values: list[int] = field(default_factory=list)
    """A collection of loss values."""

    def append(self, loss: float):
        """
        Adds a single loss value to the collection of losses.

        Args:
            loss: The loss value to add.
        """
        self._values.append(loss)

    def save(self, file_path: Path):
        """
        Writes all accumulated loss values to a file using CSV (comma separated
        values) format.

        Args:
            file_path: The path to write to.
        """
        values = array(self._values)

        with open(file_path, "w") as f:
            f.write("Loss\n")
            savetxt(f, values, delimiter=",")


def create_model_dir(name: str, output_dir: Path, parents: bool = True) -> Path:
    """
    Creates a unique directory for a single model's training output.

    The uniqueness is based on the current timestamp.  Since runs between models
    in this project are more than milliseconds apart this method is reasonable.

    Args:
        name: The model name to use.
        output_dir: The parent output directory.
        parents: Whether to automatically create any missing parent directories.

    Returns:
        A unique directory.
    """
    model_dir = output_dir / unique_name(name)

    if not model_dir.exists():
        model_dir.mkdir(parents=parents, exist_ok=False)

    return model_dir


def create_progress_bar(epochs: int, disabled: bool) -> tqdm:
    """
    Creates a :class:`tqdm` progress bar that provides a real-time display of
    completed training epochs.

    Args:
        epochs: The number of epochs to train.
        disabled: Whether to display the progress bar.

    Returns:
        A textual progress bar.
    """
    r_bar = "| {n_fmt}/{total_fmt} {unit} ({elapsed})"

    return tqdm(
        iterable=range(epochs),
        desc="Training",
        total=epochs,
        leave=False,
        disable=disabled,
        unit="epochs",
        dynamic_ncols=False,
        bar_format=f"{{l_bar}}{{bar}} {r_bar}",
        initial=0,
        position=0
    )


def save(model_dir: Path, config: Configuration, model: Model, losses: Losses):
    """
    Writes both the model weights (per library recommendation) and the
    configuration file to a timestamp-specific directory.

    Args:
        model_dir: The location of the folder to save data to.
        config: The configuration to save.
        model: The model weights to save.
        losses: A collection of per-epoch losses.
    """
    model.save(model_dir / "weights.pth")
    logger.info("Model saved to: {}.", model_dir / "weights.pth")

    losses.save(model_dir / "losses.csv")
    logger.info("Losses saved to: {}.", model_dir / "losses.csv")

    config.save(model_dir / "config.yml")
    logger.info("Configuration copied to: {}.", model_dir / "config.yml")


def parse_args() -> Namespace:
    """
    Attempts to parse one or more command line arguments.

    Returns:
        A mapping of parsed arguments and associated values, if any.
    """
    parser = ArgumentParser("cm-train", description="Train a neural cellular "
                            "automata model.")

    parser.add_argument("config_path", type=Path,
                        help="path to configuration file")
    parser.add_argument("-C", "--compile", action="store_true", default=False,
                        help="compile model to improve performance")
    parser.add_argument("-l", "--logging-level", type=str.upper, default="INFO",
                        choices=[
                            "OFF",
                            "TRACE",
                            "DEBUG",
                            "INFO",
                            "WARN",
                            "ERROR",
                            "CRITICAL"
                        ],
                        help="global logging level")
    parser.add_argument("-e", "--epochs", type=int, default=None,
                        help="override number of epochs to train")
    parser.add_argument("-n", "--name", type=str, default=None,
                        help="override model name")
    parser.add_argument("-q", "--quiet", action="store_true", default=False,
                        help="disable textual output")

    return parser.parse_args()


@logger.catch
def main() -> int:
    """
    The application entry point for this project.

    Returns:
        An exit code.
    """
    logger.remove()
    logger.add(stdout, format=LOG_FORMAT, level="ERROR", colorize=True)

    args = parse_args()
    device = choose_device()

    config = Configuration.from_file(args.config_path)
    model_dir = create_model_dir(config.name if not args.name else args.name, 
                                 config.output_dir)

    if args.logging_level != "OFF":
        logger.add(model_dir / "logs.txt", format=LOG_FORMAT,
                   level=args.logging_level, colorize=False)

        if not args.quiet:
            logger.add(TqdmSink(), format=LOG_FORMAT, level=args.logging_level,
                       colorize=True)

        logger.info("Logging enabled.")
        logger.info("Logging level set to: {}.", args.logging_level)

    logger.info("Configuration loaded from: {}.", args.config_path)
    logger.info("Model directory created at: {}.", model_dir)
    logger.info("PyTorch device chosen as: {}.", str(device))
    logger.info("Creating model, sample pool, and trainer from configuration.")

    logger.debug("Creating model and moving to device.")
    model = ConfigurationFactory.model(config).to(device)
    logger.debug("Model created successfully.")

    if args.compile:
        logger.debug("Attempting to compile model.")
        model = model.compile()
        logger.debug("Model compiled successfully.")

    logger.debug("Loading/downloading targets and creating sample pool for "
                 "training.")
    logger.debug("Training target is: {}.", config.data.target)
    pool = ConfigurationFactory.sample_pool(config)
    logger.debug("Training samples and targets created successfully.")
    
    logger.debug("Creating trainer using current model.")
    trainer = ConfigurationFactory.trainer(config, model)
    logger.debug("Trainer created successfully.")

    epochs = config.train.epochs if not args.epochs else args.epochs
    losses = Losses()
    timer = Stopwatch()

    timer.start()

    logger.info("Beginning training for {} epochs.", epochs)
    for i in create_progress_bar(epochs, args.quiet):
        logger.info("Starting epoch {} of {}.", i, epochs)

        model.train()
        loss = trainer.epoch(pool, device)
        losses.append(loss)

    timer.stop()

    logger.info("Training complete after {}.", timer.elapsed())

    save(model_dir, config, model, losses)


if __name__ == "__main__":
    exit(main())

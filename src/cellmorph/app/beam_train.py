#!/usr/bin/env python3

"""
Fully trains a neural cellular automata model using Beam.Cloud tasks to provide
GPU acceleration.

Command-line arguments:
    - "-c"/"--config-file": The path to a configuration file containing all
      necessary values and parameters to both create a model and train it.  This
      path may be either absolute or relative; however, it must obviously exist
      in order for the program to continue.
    - "-e"/"--epochs": Specifies a custom number of training epochs that
      overrides that in the configuration file.
    - "-l"/"--logging-level": The global logging level that determines which
      logging messages appear in the model-specific log file.  Any option other
      than "off" will filter messages such that if a message's level is beneath
      the global level then it will be ignored.  Please note that using "off" to
      disable logging will result in silencing *all* logging output, including
      any exceptions that are thrown in functions decorated with
      `@logger.catch` with the sole exception of `main`.
    - "-n"/"--name": Specifies a custom model name that overrides that in the
      configuration file.
    - "-t"/"--target": Specifies a custom training target that overrides that in
      the configuration file.
"""
from argparse import Namespace
from typing import Any

from beam import Image, Volume, task_queue
from loguru import logger

from cellmorph.app.cm_train import main, parse_args


@task_queue(
    cpu=2,
    memory="8Gi",
    gpu="T4",
    image=Image(python_version="python3.12", python_packages=[
        "loguru",
        "numpy",
        "pillow",
        "requests",
        "torch",
        "torchvision",
        "tqdm",
        "git+https://github.com/wbknez/cellmorph"
    ]),
    volumes=[Volume(name="model-output", mount_path="model_output")]
)
def submit(json_args: dict[str, Any]):
    """
    Creates a Beam.Cloud task that trains a single neural cellular automata
    model using GPU acceleration.

    Args:
        args: A collection of command line arguments and associated values, if
        any.
    """
    try:
        args = Namespace(**json_args)
        main(args)
        exit(0)
    except Exception as e:
        logger.exception(e)
        exit(1)


def launch():
    """
    The application entry point for this script.

    Because this script is intended to launch a Beam.Cloud task, several options
    cannot be overridden by the user.  These are:
        1. `compile` - The purpose of using Beam.Cloud is to accelerate model
           training.  As such, every model is compiled without exception.
        2. `quiet` - Because a Beam.Cloud task is non-interactive, there is no
           purpose in displaying textual output besides any error(s).  If
           progress-specific information is required, setting the logging level
           to "INFO" or "DEBUG" will accomplish this goal without clogging
           network traffic.
    """
    args = parse_args()

    object.__setattr__(args, "compile", True)
    object.__setattr__(args, "quiet", True)
    
    submit.put(vars(args))


if __name__ == "__main__":
    launch()

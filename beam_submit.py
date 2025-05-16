#!/usr/bin/env python3

# TODO: Comment me!
"""

"""
from argparse import Namespace

from beam import Image, Volume, task_queue
from loguru import logger

from cellmorph.app.cm_train import main, parse_args


@task_queue(
    cpu=2,
    memory="8Gi",
    gpu="T4",
    image=Image(python_packages=[
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
def submit(args: Namespace):
    """
    Creates a Beam task that trains a single model.

    Args:
        args: A collection of command line arguments and associated values, if
        any.
    """
    try:
        main(args)
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    args = parse_args()
    submit.put(args)

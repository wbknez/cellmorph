"""
Contains classes and functions pertaining to training the neural cellular
automata model in this project.
"""
from dataclasses import astuple, dataclass
from pathlib import Path

from loguru import logger
from PIL import Image as ImageFactory
from PIL.Image import Image
from torch import (
    Tensor,
    compile as torch_compile,
    device as torch_device,
    no_grad,
    randint
)
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from cellmorph.data import Output, SamplePool, UpdateStrategy
from cellmorph.emoji import (
    EmojiSizes,
    fetch_emoji,
    is_code,
    is_emoji,
    to_code
)
from cellmorph.image import Dimension, can_thumbnail, pad, to_tensor
from cellmorph.model import Model


def load_target(img_path: Path | str, emoji_size: int = EmojiSizes.i128,
                cache_dir: Path | None = None) -> Image:
    """
    Either obtains an emoji image from (1) an online Github repository or (2) a
    local folder cache, or reads a non-emoji image from a local file location.

    Args:
        img_path: Either the emoji or full image path to load.
        emoji_size: The initial size of the downloaded emoji image from the
        repository.
        cache_dir: The directory to cache downloaded images in; optional.

    Returns:
        An image for training.
    """
    if isinstance(img_path, str):
        if is_emoji(img_path):
            img_path = to_code(img_path)

        if is_code(img_path):
            return fetch_emoji(img_path, cache_dir=cache_dir,
                               image_size=emoji_size)
        else:
            img_path = Path(img_path).resolve()

    return ImageFactory.open(img_path)


def prepare_target(img: Image, max_size: Dimension | None = None,
                   padding: int = 16, premultiply: bool = True) -> Tensor:
    """
    Formats a target image as training input to an NCA model.

    Args:
        img: The image to manipulate.
        max_size: The new desired size.
        padding: The amount of extra space to add in all directions (top, left,
        right, and bottom).
        premultiply: Whether to premultiply the RGB components by the alpha
        channel.

    Returns:
        A formatted image as a four-dimensional PyTorch tensor.
    """
    if max_size and can_thumbnail(Dimension.from_image(img), max_size):
        img.thumbnail(astuple(max_size), ImageFactory.LANCZOS)

    if padding:
        img = pad(img, padding)

    return to_tensor(img, premultiply=premultiply)


@dataclass(frozen=True, slots=True)
class Interval:
    """
    A numeric span between two integers.
    """

    a: int
    """The minimum (smallest) value of the interval."""

    b: int
    """The maximum (largest) value of the interval."""

    def __post_init__(self):
        """
        Ensures that the minimum is less than the maximum and that the minimum
        is not negative or zero.
        """
        if self.a >= self.b:
            raise ValueError(f"Interval minimum must be less than maximum: "
                             f"{self.a} is not > {self.b}.")

        if self.a < 1:
            raise ValueError(f"Interval minimum must be positive: {self.a}.")

    def sample(self) -> int:
        """
        Computes a single random integer between this interval's bounds.

        Returns:
            A randomly chosen integer.
        """
        return randint(self.a, self.b, (1,)).item()


class Trainer:
    """
    Trains a model over one or more epochs.
    """

    _model: Model
    """A model to train."""

    _strategy: UpdateStrategy
    """How datasets are updated according to model output."""

    _steps: Interval
    """The potential number of execution steps per epoch."""

    _gradient_cutoff: float
    """The maximum gradient norm."""

    _optimizer: Adam
    """The stochastic optimizer."""

    _scheduler: MultiStepLR
    """The learning rate scheduler."""

    _batch_criterion: MSELoss
    """The loss function for an entire batch."""

    _sample_criterion: MSELoss
    """The loss function for individual samples."""

    def __init__(self, model: Model, strategy: UpdateStrategy, steps: Interval,
                 learning_rate: float, milestones: list[int], gamma: float,
                 gradient_cutoff: float):
        """
        Initializes all attributes for training.
        """
        if not model:
            raise ValueError(f"Model must be provided for training.")

        if not strategy:
            raise ValueError(f"Update strategy must be provided for training.")

        self._model = model
        self._strategy = strategy
        self._steps = steps
        self._gradient_cutoff = gradient_cutoff
        self._optimizer = Adam(model.update_rule.parameters(), lr=learning_rate)
        self._scheduler = MultiStepLR(self._optimizer, milestones, gamma=gamma)

        self._batch_criterion = MSELoss()
        self._sample_criterion = MSELoss(reduction="none")

    @property
    def gradient_cutoff(self) -> float:
        """The maximum value of a model's gradient norms during training."""
        return self._gradient_cutoff

    @property
    def learning_rate(self) -> float:
        """The current learning rate (based on the decay schedule)."""
        return self._scheduler.get_last_lr()[0]

    @property
    def model(self) -> Model:
        """The model being trained."""
        return self._model

    @property
    def steps(self) -> Interval:
        """The number of potential evaluation steps for a training period."""
        return self._steps

    @property
    def strategy(self) -> UpdateStrategy:
        """
        How a training dataset is updated according to the model
        output after a single epoch.
        """
        return self._strategy

    def epoch(self, pool: SamplePool, device: torch_device) -> float:
        """
        Completes a single epoch of training, computing both the batch and
        element-wise loss and updating the training dataset appropriately.

        Args:
            pool: A collection of training samples and target images to train
            with.
            device: The PyTorch device to use (e.g. for GPU acceleration).

        Returns:
            The total loss for a single epoch.
        """
        batch = pool.sample()
        steps = self._steps.sample()

        logger.info("Obtained {} samples from dataset.", len(batch.samples))

        samples, targets = batch.samples.to(device), batch.targets.to(device)

        logger.debug("Sent samples and targets to: {}.", device)
        logger.info("Iterating model for {} steps.", steps)
        
        for _ in range(steps):
            samples = self._model(samples)

        logger.info("Calculating batch-specific loss.")
        batch_loss = self._batch_criterion(samples[:, :4], targets[:, :4])

        logger.info("Batch loss: {}.", batch_loss.item())
        logger.info("Optimizing model for next training batch.")
        logger.debug("Beginning backwards pass.")

        self._optimizer.zero_grad()
        batch_loss.backward()

        logger.debug("Backwards pass complete.")
        logger.debug("Clipping gradient norms to {}.", self._gradient_cutoff)

        self._model.clip_gradient_norms(self._gradient_cutoff)

        logger.debug("Updating optimizer and scheduler.")

        self._optimizer.step()
        self._scheduler.step()

        logger.debug("New learning rate: {}.", self._scheduler.get_last_lr()) 

        with no_grad():
            logger.debug("Calculating sample-specific loss.")
            sample_loss = self._sample_criterion(samples[:, :4], targets[:, :4])
            sample_loss = sample_loss.mean(axis=(1, 2, 3))
            logger.debug("Sample loss: {}.", sample_loss)

        logger.info("Updating sample pool samples based on loss performance.")
        pool.update(self._strategy, Output(
            indices=batch.indices,
            x=samples.clone().detach().to("cpu"),
            loss=sample_loss.clone().detach().to("cpu")
        ))

        return batch_loss.item()

    def epochs(self, pool: SamplePool, epochs: int,
               device: torch_device) -> list[float]:
        """
        Completes multiple epochs of training, computing both the batch and
        element-wise loss and updating the training dataset appropriately.

        Args:
            pool: A collection of training samples and target images to train
            with.
            epochs: The number of epochs to train.
            device: The PyTorch device to use (e.g. for GPU acceleration).

        Returns:
            The total loss for each epoch as a list.
        """
        losses = []

        for i in range(epochs):
            loss = self.epoch(pool, device)
            losses.append(loss)

        return losses

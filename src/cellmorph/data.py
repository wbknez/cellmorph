"""
Contains classes and functions to facilitate training using single-image
datasets that implement all three "experiments", or techniques, in the original
paper.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple, override

from torch import Tensor, equal, linspace, zeros
from torch.distributions.uniform import Uniform
from torch.utils.data import DataLoader, RandomSampler, Sampler, TensorDataset

from cellmorph.image import Dimension


def damage_mask(sample_count: int, size: Dimension) -> Tensor:
    """
    Creates one or more circular masks that are intended to render portions of a
    training image invisible.

    Conceptually, this mask "damages" an image by removing one or more
    significant portions of it from a model.

    Args:
        sample_count: The number of masks to create.
        size: The maximum size of the damage mask.

    Returns:
        One or more damage masks as a single Tensor.
    """
    theta = linspace(-1.0, 1.0, size.width)[None, None, :]
    phi = linspace(-1.0, 1.0, size.height)[None, :, None]

    center = Uniform(-0.5, 0.5).sample((2, sample_count, 1, 1))
    r = Uniform(0.1, 0.4).sample((sample_count, 1, 1))

    x = (theta - center[0]) / r
    y = (phi - center[1]) / r

    mask = ((x * x + y * y) < 1.0).float().unsqueeze(1)

    return 1.0 - mask


def empty_seed(state_channels: int, size: Dimension, sample_count: int = 1,
               pos: Position | None = None) -> Tensor:
    """
    Creates a new empty tensor for model consumption with a single active
    automata.

    The active automata is always in the center (width and height divided by
    two).

    Args:
        state_channels: The number of state channels per pixel.
        size: The dimension of the seed image to generate in pixels.
        sample_count: The number of automata to generate seeds for.
        pos: The location of the single automata to activate.

    Returns:
        A new model seed as a four-dimensional tensor.
    """
    if not pos:
        pos = Position(size.width // 2, size.height // 2)

    shape = (sample_count, state_channels, size.height, size.width)
    seed = zeros(shape).float()

    seed[:, 3:, pos.y, pos.x] = 1.0

    return seed


@dataclass(frozen=True, slots=True)
class Position:
    """
    A specific location in two-dimensional Cartesian space.
    """

    x: int
    """The x-axis coordinate."""

    y: int
    """The y-axis coordinate."""

    def __post_init__(self):
        if self.x < 0:
            raise ValueError(f"X-axis coordinate cannot be negative: {self.x}.")

        if self.y < 0:
            raise ValueError(f"Y-axis coordinate cannot be negative: {self.y}.")


class Sample(NamedTuple):
    """
    A single training sample from a larger dataset.
    """

    index: int | Tensor
    """The index of a sample in a larger dataset."""

    sample: Tensor
    """A sample to train on."""

    target: Tensor
    """A target image to train against."""


@dataclass(frozen=True, slots=True)
class Batch:
    """
    A subset of randomly chosen samples from a dataset.
    """

    indices: Tensor
    """A collection of training sample indices chosen from a larger dataset."""

    samples: Tensor
    """A collection of training samples chosen from a larger dataset."""

    targets: Tensor
    """A collection of training targets from a larger dataset."""

    def __post_init__(self):
        """
        Ensures that each field is valid and usable in model training.
        """
        if self.indices is None or self.indices.numel() == 0:
            raise ValueError("No indices provided in batch.")

        if self.samples is None or self.samples.numel() == 0:
            raise ValueError("No samples provided in batch.")

        if self.targets is None or self.targets.numel() == 0:
            raise ValueError("No targets provided in batch.")


@dataclass(frozen=True, slots=True)
class Output:
    """
    The output from a single model step.
    """

    indices: Tensor
    """A collection of sample indices from a larger dataset."""

    x: Tensor
    """The generated model output."""

    loss: Tensor
    """The computed loss per individual sample."""

    def __post_init__(self):
        if self.indices is None or self.indices.numel() == 0:
            raise ValueError("No indices provided.")

        if self.x is None or self.x.numel() == 0:
            raise ValueError("No output provided.")

        if self.loss is None or self.loss.numel() == 0:
            raise ValueError("No loss provided.")


class IndexingDataset(TensorDataset):
    """
    A :class:`TensorDataset` that returns the requested indices in addition to
    the corresponding samples and targets.
    """

    def __init__(self, sample_count: int, state_channels: int,
                 targets: Tensor, start: Position | None = None):
        """
        Creates an initial training dataset consisting of empty model seeds.

        Args:
            sample_count: The amount of training samples to create.
            state_channels: The state space size per automata.
            target: The target image(s) to train on.
            start: The starting position on an empty seed.

        Raises:
            ValueError: If the number of training targets is not the same as the
            number of samples.
        """
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(0)

        if len(targets) == 1:
            targets = targets.repeat((sample_count, 1, 1, 1))

        if not len(targets) == sample_count:
            raise ValueError("The number of targets is different than the "
                             "number of samples.")

        size = Dimension.from_tensor(targets)

        samples = empty_seed(state_channels, size, sample_count, start)

        super().__init__(samples, targets)

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, IndexingDataset):
            return equal(self.samples, other.samples) and \
                equal(self.targets, other.targets)

        return NotImplemented

    @override
    def __getitem__(self, index: int | Tensor) -> Sample:
        """
        Returns both a new model seed and the current training image for a
        particular batch.

        Args:
      index: The sample index to obtain.

        Returns:
            A tuple containing both a training image and target image.
        """
        return Sample(index, self.tensors[0][index], self.tensors[1][index])

    @override
    def __len__(self) -> int:
        """
        Returns the number of training images this dataset can provide.

        Returns:
            The number of available training images.
        """
        return len(self.tensors[0])

    @override
    def __ne__(self, other: object) -> bool:
        return not self == other

    @property
    def samples(self) -> Tensor:
        """The current starting states to train with."""
        return self.tensors[0]

    @property
    def targets(self) -> Tensor:
        """The final target states to train against."""
        return self.tensors[1]


class UpdateStrategy(ABC):
    """
    Updates one or more samples in a dataset on a per batch basis.
    """

    __slots__ = ("_initial_state")

    _initial_state: Tensor
    """A single, distinct seed to apply as an initial starting state."""

    def __init__(self, initial_state: Tensor):
        """
        Initializes the default state that should be used.

        Args:
            initial_state: The unique state to use as a default.

        Raises:
            ValueError: If `initial_state` is not valid or empty.
        """
        if initial_state is None or initial_state.numel() == 0:
            raise ValueError("Initial state is not valid for strategy.")

        self._initial_state = initial_state

    @abstractmethod
    def apply(self, output: Output, ds: IndexingDataset):
        """
        Updates a small portion of a dataset with model output based on
        performance (loss).

        Args:
            output: The model output to utilize.
            ds: The dataset to update.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a unique string identifier used for converting a strategy to a
        YAML value.

        Returns:
            A unique standard string identifier.
        """
        pass


class GrowthStrategy(UpdateStrategy):
    """
    An :class:`UpdateStrategy` that resets each sample with an initial state.

    This strategy implements the first "experiment" in the original paper.
    """

    __slots__ = ()

    def __init__(self, initial_state: Tensor):
        super().__init__(initial_state)

    @override
    def apply(self, output: Output, ds: IndexingDataset):
        """
        Re-initializes all samples in a batch to some initial state.

        Args:
            output: The model output to utilize.
            ds: The dataset to update.
        """
        ds.samples[output.indices][:] = self._initial_state

    @override
    def __str__(self) -> str:
        return "Growth"


class PersistentStrategy(UpdateStrategy):
    """
    An :class:`UpdateStrategy` that overwrites samples with their model outputs
    and resets the lowest performing (highest loss) sample to an initial state.

    This strategy implements the second "experiment" in the original paper.  By
    retaining well-performing outputs the model is better able to learn to both
    grow and stop (i.e. persist) instead of continuous infinite growth.
    """

    __slots__ = ("_reset_count")

    _reset_count: int
    """
    The number of lowest performing (highest loss) samples to reset to an
    initial state.
    """

    def __init__(self, initial_state: Tensor, reset_count: int = 1):
        """
        Initializes all attributes as appropriate.

        Args:
            initial_state: The unique state to use as a default.
            reset_count: The number of lowest performing samples to reset.

        Raises:
            ValueError: If `initia_state` is not valid, `reset_count` is less
            than zero.
        """
        super().__init__(initial_state)

        if reset_count < 0:
            raise ValueError(f"Reset count cannot be negative: {reset_count}.")

        self._reset_count = reset_count

    @override
    def apply(self, output: Output, ds: IndexingDataset):
        """
        Overwrites all samples with their model outputs except the lowest
        performing (highest loss) sample which is re-initialzed to some initial
        state.

        Args:
            output: The model output to utilize.
            ds: The dataset to update.
        """
        ordered = output.indices[output.loss.argsort(descending=True)]

        worst = ordered[:self._reset_count]

        ds.samples[output.indices] = output.x
        ds.samples[worst] = self._initial_state

    @override
    def __str__(self) -> str:
        return "Persistent"


class RegenerativeStrategy(PersistentStrategy):
    """
    An :class:`UpdateStrategy` that overwrites samples with their model outputs,
    resets the lowest performing (highest loss) sample to an initial state, and
    applies a damage mask to the highest performers (lowest loss).

    This strategy implements the third "experiment" in the original paper.  By
    applying damage masks to the best outputs, the model is able to learn to
    repair those damaged regions.
    """

    __slots__ = ("_damage_count")

    _damage_count: int
    """
    The number of highest performing (lowest loss) samples to both retain and
    apply a damage mask to in order to foster regenerative properties.
    """

    def __init__(self, initial_state: Tensor, reset_count: int = 1,
                 damage_count: int = 3):
        """
        Initializes all attributes as appropriate.

        Args:
            initial_state: The unique state to use as a default.
            reset_count: The number of lowest performing samples to reset.
            damage_count: The number of highest-performing samples to apply
            damage masks to.

        Raises:
            ValueError: If `initia_state` is not valid, `reset_count` is less
            than zero, or `damage_mask` is less than zero.
        """
        super().__init__(initial_state, reset_count)

        if damage_count < 0:
            raise ValueError(f"Damage count must be positive: {damage_count}.")

        self._damage_count = damage_count

    @override
    def apply(self, output: Output, ds: IndexingDataset):
        """
        Overwrites all samples with their model outputs except the lowest
        performing (highest loss) sample which is re-initialzed to some initial
        state; in addition, the highest performing (lowest loss) outputs are
        altered with damage masks.

        Args:
            output: The model output to utilize.
            ds: The dataset to update.
        """
        ordered = output.indices[output.loss.argsort(descending=True)]
        size = Dimension.from_tensor(output.x[0])

        worst = ordered[:self._reset_count]
        best = ordered[-self._damage_count:]

        ds.samples[output.indices] = output.x
        ds.samples[worst] = self._initial_state
        ds.samples[best] *= damage_mask(len(best), size)

    @override
    def __str__(self) -> str:
        return "Regenerative"


class SamplePool(DataLoader):
    """
    A :class:`DataLoader` that provides training data as indexed random batches.
    """
    
    def __init__(self, ds: IndexingDataset, batch_size: int, 
                 sampler: Sampler | None = None):
        """
        Initializes the sample pool with a specific batch size.

        Args:
            ds: A collection of training samples and associated targets.
            batch_size: The number of samples per random batch.
            sampler: How samples are selected per batch.
        """
        if not sampler:
            sampler = RandomSampler(range(len(ds)))

        super().__init__(
            dataset=ds,
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
            pin_memory=False
        )

    @override
    def __iter__(self) -> Iterator[Batch]:
        """
        Iterates over a dataset and provides batches of sample data for
        training.

        Returns:
            A single batch of indexed training data.
        """
        it = super().__iter__()

        for (indices, samples, target) in it:
            yield Batch(indices, samples, target)

    def sample(self) -> Batch:
        """
        Selects a random batch from this sample pool.

        Returns:
            A random batch.
        """
        return next(iter(self))

    def update(self, strategy: OutputStrategy, output: Output):
        """
        Updates the underlying dataset based on model output according to a
        specific strategy.

        Args:
            strategy: The update strategy to use.
            output: The model output to incorporate.
        """
        strategy.apply(output, self.dataset)

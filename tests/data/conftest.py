"""
Contains common fixtures and utilites for data-specific unit testing.
"""
from pytest import fixture
from torch import Tensor, rand, randperm
from torch.distributions.uniform import Uniform

from cellmorph.data import Batch, IndexingDataset, Output


@fixture(scope="function")
def ds(sample_count: int, state_channels: int, height: int,
       width: int) -> IndexingDataset:
    """
    Creates an :class:`IndexingDataset` that is filled with both random samples
    and random targets (RGBA images).

    Args:
        sample_count: The number of samples to generate.
        state_channels: The number of states per automata.
        height: The height of the simulation space.
        width: The width of the simulation space.

    Returns:
        A dataset filled with random data.
    """
    targets = rand((sample_count, 4, height, width)).float()

    return IndexingDataset(sample_count, state_channels, targets)


@fixture(scope="function")
def indices(index_count: int, sample_count: int) -> Tensor:
    """
    Creates a list of unique indices for a specific number of samples.

    Args:
        index_count: The number of indices to generate.
        sample_count: The number of samples that denotes the maximum index
        value.

    Returns:
        A tensor of random integer indices.
    """
    return randperm(sample_count)[:index_count]


@fixture(scope="function")
def batch(indices: Tensor, ds: IndexingDataset) -> Batch:
    """
    Creates a single batch filled with randomized indices and data.

    Args:
        indices: The randomized indices to use.
        ds: The randomized dataset to draw from.

    Returns:
        A random batch.
    """
    return Batch(indices, ds.samples[indices], ds.targets[indices])


@fixture(scope="function")
def loss(indices: Tensor) -> Tensor:
    """
    Creates a :class:`Tensor` filled with random floating-point values.

    Args:
        indices: The indices to use.

    Returns:
        A random tensor.
    """
    return rand(indices.shape).float()


@fixture(scope="function")
def output(batch: Batch, loss: Tensor) -> Tensor:
    """
    Creates an :class:`Output` filled with randomized indices and output values.

    Args:
        batch: A randomized batch to use.
        loss: A set of randomized loss values to use.

    Returns:
        A randomized output.
    """
    x = rand(batch.samples.shape).float()

    return Output(batch.indices, x, loss)

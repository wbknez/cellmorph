"""
Ensures that obtaining data and target samples from this project's custom
dataset works as expected.
"""
from typing import override

from numpy.random import Generator
from pytest import fixture, mark
from torch import Size, Tensor, equal, from_numpy

from cellmorph.training import Batch, Output, Position, SingleTargetDataset


class NullDataset(SingleTargetDataset):
    def __init__(self, sample_count: int, state_channels: int, target: Tensor,
                 start: Position | None = None):
        super().__init__(sample_count, state_channels, target, start)

    @override
    def update_batch(self, indices: Tensor, output: Output):
        pass


@fixture(scope="function")
def ds(sample_count: int, state_channels: int, rng: Generator) -> NullDataset:
    target = from_numpy(rng.random((1, state_channels, 72, 72))).float()
    return NullDataset(sample_count, state_channels, target)


@fixture(scope="function")
def indices(sample_count: int, rng: Generator) -> Tensor:
    return from_numpy(rng.choice(sample_count, 16, replace=False))


class TestSingleTargetDataset:
    """
    Test suite for :class:`SingleTargetDataset`.
    """

    @mark.parametrize('sample_count, state_channels, expected', [
        (1024, 16, Size((1024, 16, 72, 72))),
        (512, 64, Size((512, 64, 72, 72))),
        (32, 32, Size((32, 32, 72, 72)))
    ])
    def test_dataset_expands_target_to_data_size(
        self, ds: NullDataset, expected: Size
    ):
        assert ds._data.size() == expected
        assert ds._target.size() == expected

    @mark.parametrize('sample_count, state_channels, expected', [
        (1024, 16, 1024),
        (512, 64, 512),
        (32, 32, 32)
    ])
    def test_dataset_length_returns_sample_count(
        self, ds: NullDataset, expected: int
    ):
        assert len(ds) == expected
        assert len(ds._data) == len(ds._target)

    @mark.parametrize('sample_count, state_channels, expected', [
        (1024, 16, 1024),
        (512, 64, 512),
        (32, 32, 32)
    ])
    def test_dataset_repeats_target_data_correctly(
        self, ds: NullDataset, expected: int
    ):
        assert len(ds._target) == expected

    @mark.parametrize('sample_count, state_channels', [
        (1024, 16),
        (512, 64),
        (32, 32)
    ])
    def test_dataset_get_item_returns_appropriate_partitions(
        self, indices: Tensor, ds: NullDataset
    ):
        expected = Batch(indices, ds._data[indices], ds._target[indices])
        result = ds[indices]

        assert equal(result.indices, expected.indices)

"""
Ensures that obtaining data and target samples from one of this project's custom
dataset - regenerative - works as expected.
"""
from typing import override

from numpy.random import Generator
from pytest import fixture, mark
from torch import Size, Tensor, equal, from_numpy, manual_seed

from cellmorph.image import Dimension
from cellmorph.training import (
    Batch,
    Output,
    Position,
    RegenerativeDataset,
    damage_mask
)
from cellmorph.utils import random_bytes


@fixture(scope="function")
def ds(sample_count: int, state_channels: int,
       rng: Generator) -> RegenerativeDataset:
    target = from_numpy(rng.random((1, state_channels, 72, 72))).float()
    return RegenerativeDataset(sample_count, state_channels, target)


@fixture(scope="function")
def indices(sample_count: int, rng: Generator) -> Tensor:
    return from_numpy(rng.choice(sample_count, 16, replace=False))


@fixture(scope="function")
def loss(indices: Tensor, rng: Generator) -> Tensor:
    return from_numpy(rng.random(len(indices))).float()


class TestRegenerativeDataset:
    """
    Test suite for :class:`RegenerativeDataset`.
    """

    @mark.parametrize('sample_count, state_channels, expected', [
        (1024, 16, Size((1024, 16, 72, 72))),
        (512, 64, Size((512, 64, 72, 72))),
        (32, 32, Size((32, 32, 72, 72)))
    ])
    def test_dataset_expands_target_to_data_size(
        self, ds: RegenerativeDataset, expected: Size
    ):
        assert ds._data.size() == expected
        assert ds._target.size() == expected

    @mark.parametrize('sample_count, state_channels, expected', [
        (1024, 16, 1024),
        (512, 64, 512),
        (32, 32, 32)
    ])
    def test_dataset_length_returns_sample_count(
        self, ds: RegenerativeDataset, expected: int
    ):
        assert len(ds) == expected
        assert len(ds._data) == len(ds._target)

    @mark.parametrize('sample_count, state_channels, expected', [
        (1024, 16, 1024),
        (512, 64, 512),
        (32, 32, 32)
    ])
    def test_dataset_repeats_target_data_correctly(
        self, ds: RegenerativeDataset, expected: int
    ):
        assert len(ds._target) == expected

    @mark.parametrize('sample_count, state_channels', [
        (1024, 16),
        (512, 64),
        (32, 32)
    ])
    def test_dataset_get_item_returns_appropriate_partitions(
        self, indices: Tensor, ds: RegenerativeDataset
    ):
        expected = Batch(indices, ds._data[indices], ds._target[indices])
        result = ds[indices]

        assert equal(result.indices, expected.indices)

    @mark.parametrize('sample_count, state_channels', [
        (1024, 16),
        (512, 64),
        (32, 32)
    ])
    def test_dataset_update_batch_damages_top_three_performers(
        self, indices: Tensor, ds: RegenerativeDataset, loss: Tensor
    ):
        batch = ds[indices]
        output = Output(batch.samples.clone().detach() * 2.5, loss)

        seed = random_bytes()

        ordered = indices[loss.argsort(descending=True)]
        size = Dimension.from_tensor(ds._target)

        manual_seed(seed)

        ds.update_batch(indices, output)

        manual_seed(seed)

        data_copy = ds._data.clone().detach()
        data_copy[ordered[:1]] = ds._empty_seed
        data_copy[ordered[-3:]] *= damage_mask(3, size)

        expected = data_copy[indices].clone().detach()
        result = ds[indices].samples

        assert equal(result, expected)

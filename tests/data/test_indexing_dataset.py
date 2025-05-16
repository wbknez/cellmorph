"""
Ensures that obtaining data and target samples from a custom dataset works as
expected.
"""
from pytest import mark, raises
from torch import Tensor, equal, rand, zeros_like

from cellmorph.data import IndexingDataset, Sample


class TestIndexingDataset:
    """
    Test suite for :class:`IndexingDataset`.
    """
    
    @mark.parametrize('sample_count, state_channels, target_count, height, width', [
        (40, 16, 19, 40, 40)
    ])
    def test_constructor_raises_if_target_count_is_not_the_same_as_samples(
        self, sample_count: int, state_channels: int, target: Tensor
    ):
        with raises(ValueError):
            IndexingDataset(sample_count, state_channels, target)

    @mark.parametrize('sample_count, state_channels, target', [
        (40, 16, rand((4, 40, 40))),
        (97, 18, rand((4, 16, 32))),
    ])
    def test_constructor_expands_targets_to_match_samples_if_count_is_one(
        self, sample_count: int, state_channels: int, target: Tensor
    ):
        ds = IndexingDataset(sample_count, state_channels, target)

        expected = (sample_count, 4, target.shape[-2], target.shape[-1])

        assert ds.targets.shape == expected

    @mark.parametrize('sample_count, state_channels, target_count, height, width', [
        (40, 37, 40, 41, 23)
    ])
    def test_constructor_creates_sample_tensors_of_correct_size(
        self, sample_count: int, state_channels: int, target: Tensor
    ):
        ds = IndexingDataset(sample_count, state_channels, target)

        expected = (sample_count, 37, 41, 23)

        assert ds.samples.shape == expected

    @mark.parametrize('sample_count, state_channels, target_count, height, width', [
        (90, 17, 90, 44, 73)
    ])
    def test_len_returns_number_of_samples(
        self, sample_count: int, state_channels: int, target: Tensor
    ):
        ds = IndexingDataset(sample_count, state_channels, target)

        assert len(ds) == sample_count

    @mark.parametrize('sample_count, state_channels, target_count, height, width', [
        (1024, 30, 1024, 15, 12)
    ])
    def test_iteration_produces_correct_samples_by_index(
        self, sample_count: int, state_channels: int, target: Tensor
    ):
        ds = IndexingDataset(sample_count, state_channels, target)

        for i in range(sample_count):
            expected = Sample(i, ds.samples[i], ds.targets[i])
            result = ds[i]

            assert result.index == expected.index
            assert equal(result.sample, expected.sample)
            assert equal(result.target, expected.target)

    @mark.parametrize('sample_count, state_channels, target_count, height, width', [
        (1024, 30, 1024, 15, 12)
    ])
    def test_eq_evaluates_correctly(
        self, sample_count: int, state_channels: int, target: Tensor
    ):
        expected = IndexingDataset(sample_count, state_channels, target)
        result = IndexingDataset(sample_count, state_channels, target)

        assert result == expected

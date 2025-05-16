"""
Ensures that obtaining batches of samples from a custom dataloader works
correctly.
"""
from torch import empty, equal, rand, randperm
from torch.utils.data import RandomSampler

from cellmorph.data import (
    GrowthStrategy,
    IndexingDataset,
    Output,
    PersistentStrategy,
    SamplePool,
    empty_seed
)
from cellmorph.image import Dimension


class TestSamplePool:
    """
    Test suite for :class:`SamplePool`.
    """

    def test_default_sampler_is_random_sampler(self):
        targets = rand((1024,  4, 72, 72))

        ds = IndexingDataset(1024, 16, targets)
        pool = SamplePool(ds, batch_size=8, sampler=None)

        assert isinstance(pool.sampler, RandomSampler)
        assert pool.sampler.data_source == ds

    def test_full_iteration_produces_unique_indices(self):
        targets = rand((1024, 4, 72, 72))

        ds = IndexingDataset(1024, 16, targets)
        pool = SamplePool(ds, batch_size=16)

        expected = set(range(1024))
        result = set()

        for batch in pool:
            result.update([index.item() for index in batch.indices])

        assert result == expected

    def test_sample_iteration_produces_repeated_indices(self):
        targets = rand((1024, 4, 72, 72))

        ds = IndexingDataset(1024, 16, targets)
        pool = SamplePool(ds, batch_size=16)

        min_iterations = (1024 // 16) + 1
        indices = list()

        for _ in range(min_iterations):
            batch = pool.sample()
            indices.extend([index.item() for index in batch.indices])

        uniques = set(indices)
        result = [indices.count(elem) for elem in uniques]

        assert sum(result) > 1024

    def test_update_with_growth_strategy(self):
        targets = rand((1024, 4, 72, 72))
        initial_state = empty_seed(16, Dimension(72, 72))

        ds = IndexingDataset(1024, 16, targets)
        pool = SamplePool(ds, batch_size=16)

        strategy = GrowthStrategy(initial_state)
        output = Output(
            indices=randperm(1024)[:50],
            x=rand((50, 16, 72, 72)),
            loss=rand(50)
        )

        pool.update(strategy, output)

        expected = initial_state.repeat((50, 1, 1, 1))
        result = ds.samples[output.indices]

        assert equal(result, expected)

    def test_update_with_persistent_strategy(self):
        targets = rand((1024, 4, 72, 72))
        initial_state = empty_seed(16, Dimension(72, 72))

        ds = IndexingDataset(1024, 16, targets)
        pool = SamplePool(ds, batch_size=16)

        strategy = PersistentStrategy(initial_state, reset_count=2)
        output = Output(
            indices=randperm(1024)[:50],
            x=rand((50, 16, 72, 72)),
            loss=rand(50)
        )

        pool.update(strategy, output)

        ordered = output.indices[output.loss.argsort(descending=True)]
        samples = initial_state.repeat((1024, 1, 1, 1))

        samples[output.indices] = output.x
        samples[ordered[:2]] = initial_state

        expected = samples[output.indices]
        result = ds.samples[output.indices]

        assert equal(result, expected)

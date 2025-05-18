"""
Ensures that updating a dataset based on loss with a regenerative strategy works
as expected.
"""
from pytest import mark, raises
from torch import Tensor, equal, manual_seed, zeros_like

from cellmorph.data import (
    Batch,
    Dimension,
    RegenerativeStrategy,
    IndexingDataset,
    Output,
    damage_mask,
    empty_seed
)
from cellmorph.utils import random_bytes


class TestRegenerativeStrategy:
    """
    Test suite for :class:`RegenerativeStrategy`.
    """

    @mark.parametrize('initial_state', [ None, Tensor([]).float() ])
    def test_init_raises_if_initial_state_is_not_valid(
        self, initial_state: Tensor
    ):
        with raises(ValueError):
            RegenerativeStrategy(initial_state, 1)

    @mark.parametrize('reset_count', [ -1, -2 ])
    def test_init_raises_if_reset_count_is_negative(self, reset_count: int):
        initial_state = empty_seed(16, Dimension(72, 72))

        with raises(ValueError):
            RegenerativeStrategy(initial_state, reset_count)

    @mark.parametrize('sample_count,state_channels,height,width,index_count, reset_count,damage_count', [
        (1024, 16, 72, 72, 50, 1, 3),
        (512, 32, 24, 24, 100, 3, 10),
        (256, 32, 24, 24, 200, 40, 60),
    ])
    def test_apply_resets_all_samples(
        self, output: Output, ds: IndexingDataset, reset_count: int,
        damage_count:int
    ):
        seed = random_bytes()

        size = Dimension.from_tensor(ds.samples[0])
        initial_state = empty_seed(ds.samples[0].shape[0], size)

        strategy = RegenerativeStrategy(initial_state, reset_count,
                                        damage_count)
        ordered = output.indices[output.loss.argsort(descending=True)]

        worst = ordered[:reset_count]
        best = ordered[-damage_count:]

        samples = zeros_like(ds.samples)
        samples[:] = ds.samples[:]

        manual_seed(seed)
        strategy.apply(output, ds)

        manual_seed(seed)
        samples[output.indices] = output.x
        samples[worst] = initial_state
        samples[best] *= damage_mask(len(best), size)

        expected = samples[output.indices]
        result = ds.samples[output.indices]

        assert equal(result, expected)

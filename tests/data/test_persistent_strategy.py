"""
Ensures that updating a dataset based on loss with a persistent strategy works
as expected.
"""
from pytest import mark, raises
from torch import Tensor, equal, zeros_like

from cellmorph.data import (
    Batch,
    PersistentStrategy,
    IndexingDataset,
    Output,
    empty_seed
)
from cellmorph.image import Dimension


class TestPersistentStrategy:
    """
    Test suite for :class:`PersistentStrategy`.
    """

    @mark.parametrize('initial_state', [ None, Tensor([]).float() ])
    def test_init_raises_if_initial_state_is_not_valid(
        self, initial_state: Tensor
    ):
        with raises(ValueError):
            PersistentStrategy(initial_state, 1)

    @mark.parametrize('reset_count', [ -1, -2 ])
    def test_init_raises_if_reset_count_is_negative(self, reset_count: int):
        initial_state = empty_seed(16, Dimension(72, 72))

        with raises(ValueError):
            PersistentStrategy(initial_state, reset_count)

    @mark.parametrize('sample_count,state_channels,height,width,index_count, reset_count', [
        (1024, 16, 72, 72, 50, 1),
        (512, 32, 24, 24, 100, 3),
        (256, 32, 24, 24, 200, 40),
    ])
    def test_apply_resets_all_samples(
        self, output: Output, ds: IndexingDataset, reset_count: int
    ):
        size = Dimension.from_tensor(ds.samples[0])
        initial_state = empty_seed(ds.samples[0].shape[0], size)

        strategy = PersistentStrategy(initial_state, reset_count)
        ordered = output.indices[output.loss.argsort(descending=True)]

        worst = ordered[:reset_count]

        strategy.apply(output, ds)

        samples = zeros_like(ds.samples)

        samples[output.indices] = output.x
        samples[worst] = initial_state

        expected = samples[output.indices]
        result = ds.samples[output.indices]

        assert equal(result, expected)

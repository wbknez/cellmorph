"""
Ensures that updating a dataset based on loss with a growth strategy works as
expected.
"""
from pytest import mark, raises
from torch import Tensor, equal

from cellmorph.data import (
    Batch,
    GrowthStrategy,
    IndexingDataset,
    Output,
    empty_seed
)
from cellmorph.image import Dimension


class TestGrowthStrategy:
    """
    Test suite for :class:`GrowthStrategy`.
    """

    @mark.parametrize('initial_state', [ None, Tensor([]).float() ])
    def test_init_raises_if_initial_state_is_not_valid(
        self, initial_state: Tensor
    ):
        with raises(ValueError):
            GrowthStrategy(initial_state)

    @mark.parametrize('sample_count,state_channels,height,width,index_count', [
        (1024, 16, 72, 72, 50),
        (512, 32, 24, 24, 100),
        (256, 32, 24, 24, 200),
    ])
    def test_apply_resets_all_samples(
        self, output: Output, ds: IndexingDataset
    ):
        size = Dimension.from_tensor(ds.samples[0])
        initial_state = empty_seed(ds.samples[0].shape[0], size)

        strategy = GrowthStrategy(initial_state)

        strategy.apply(output, ds)

        expected = strategy._initial_state.repeat(
            (len(output.indices), 1, 1, 1)
        )
        result = ds.samples[output.indices]

        assert equal(result, expected)

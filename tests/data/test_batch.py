"""
Ensures that batch initialization verifies input arguments correctly.
"""
from pytest import mark, raises
from torch import Tensor, empty, rand, randint

from cellmorph.data import Batch


class TestBatch:
    """
    Test suite for :class:`Batch`.
    """

    @mark.parametrize('indices', [ None, empty(0) ])
    def test_batch_raises_if_indices_are_none(self, indices: Tensor):
        with raises(ValueError):
            Batch(indices, rand(10, 16, 72, 72), rand(1, 16, 72, 72))

    @mark.parametrize('samples', [ None, empty(0) ])
    def test_batch_raises_if_samples_are_none(self, samples: Tensor):
        with raises(ValueError):
            Batch(randint(1, 5, (5,)), samples, rand(1, 16, 72, 72))

    @mark.parametrize('target', [ None, empty(0) ])
    def test_batch_raises_if_target_is_none(self, target: Tensor):
        with raises(ValueError):
            Batch(randint(1, 5, (10,)), rand(10, 16, 72, 72), target)

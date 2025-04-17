"""
Ensures that batch initialization verifies input arguments correctly.
"""
from pytest import mark, raises
from torch import Tensor, empty, rand, randint

from cellmorph.training import Batch


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

    @mark.parametrize('indices, samples', [
        (randint(0, 5, (10,)), rand(9, 16, 72, 72)),
        (randint(0, 23, (23,)), rand(26, 42, 99, 99))
    ])
    def test_batch_raises_if_indices_shape_is_not_the_same_as_target(
        self, indices: Tensor, samples: Tensor
    ):
        with raises(ValueError):
            Batch(indices, samples, rand(1, 16, 72, 72))

    @mark.parametrize('samples, target', [
        (rand(19, 16, 72, 72), rand(1, 15, 72, 72)),
        (rand(19, 16, 72, 72), rand(1, 16, 71, 72)),
        (rand(19, 16, 72, 72), rand(1, 16, 72, 73)),
        (rand(19, 11, 72, 72), rand(1, 16, 72, 72)),
    ])
    def test_batch_raises_if_samples_size_is_not_the_same_as_target(
        self, samples: Tensor, target: Tensor
    ):
        size = len(samples)

        with raises(ValueError):
            Batch(randint(0, size, (size,)), samples, target)

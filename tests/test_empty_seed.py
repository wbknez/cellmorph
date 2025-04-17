"""
Ensures that model seed creation works as expected.
"""
from numpy import float32, zeros
from pytest import mark, raises
from torch import equal, tensor

from cellmorph.image import Dimension
from cellmorph.training import Position, empty_seed


class TestEmptySeed:
    """
    Test suite for :meth:`empty_seed`.
    """

    @mark.parametrize('size', [
        Dimension(40, 40),
        Dimension(10, 50),
        Dimension(75, 15)
    ])
    def test_model_seed_only_has_one_active_channel( self, size: Dimension):
        expected = zeros((1, 16, size.height, size.width), dtype=float32)
        result = empty_seed(size, 16)

        expected[:, 3:, size.height // 2, size.width // 2] = 1.0

        assert equal(result, tensor(expected))

    @mark.parametrize('size, pos', [
        (Dimension(40, 40), Position(5, 37)),
        (Dimension(10, 50), Position(9, 11)),
        (Dimension(75, 15), Position(0, 0)),
        (Dimension(75, 15), Position(74, 14))
    ])
    def test_model_seed_only_has_one_active_channel_at_x_and_y(
        self, size: Dimension, pos: Position
    ):
        expected = zeros((1, 16, size.height, size.width), dtype=float32)
        result = empty_seed(size, 16, pos=pos)

        expected[:, 3:, pos.y, pos.x] = 1.0

        assert equal(result, tensor(expected))

    @mark.parametrize('sample_count, size', [
        (49, Dimension(40, 40)),
        (1, Dimension(10, 50)),
        (15, Dimension(75, 15))
    ])
    def test_model_seed_shape_is_correct(
        self, sample_count: int, size: Dimension
    ):
        seed = empty_seed(size, 16, sample_count=sample_count)

        assert seed.shape == (sample_count, 16, size.height, size.width)

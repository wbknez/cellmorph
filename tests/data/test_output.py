"""
Ensures that model output is checked for shape correctness.
"""
from pytest import mark, raises
from torch import Tensor, empty, rand

from cellmorph.data import Output


class TestOutput:
    """
    Test suite for :class:`Output`.
    """

    @mark.parametrize('values', [ None, empty(0) ])
    def test_output_raises_if_indices_are_not_valid( self, values: Tensor):
        x = rand(5)
        loss = rand(5)

        with raises(ValueError):
            Output(values, x, loss)

    @mark.parametrize('values', [ None, empty(0) ])
    def test_output_raises_if_x_is_not_valid(self, values: Tensor):
        indices = rand(5)
        loss = rand(5)

        with raises(ValueError):
            Output(indices, values, loss)

    @mark.parametrize('values', [ None, empty(0) ])
    def test_output_raises_if_loss_is_not_valid( self, values: Tensor):
        indicesloss = rand(5)
        x = rand(5)

        with raises(ValueError):
            Output(values, x, values)

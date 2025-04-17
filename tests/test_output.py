"""
Ensures that model output is checked for shape correctness.
"""
from pytest import mark, raises
from torch import Tensor, empty, rand

from cellmorph.training import Output


class TestOutput:
    """
    Test suite for :class:`Output`.
    """

    @mark.parametrize('x', [ None, empty(0) ])
    def test_output_raises_if_x_is_not_valid(self, x: Tensor):
        loss = rand(5)

        with raises(ValueError):
            Output(x, loss)

    @mark.parametrize('loss', [ None, empty(0) ])
    def test_output_raises_if_loss_is_not_valid(self, loss: Tensor):
        x = rand(7, 16, 72, 72)

        with raises(ValueError):
            Output(x, loss)

    def test_output_raises_if_x_and_loss_shape_are_not_the_same(self):
        x = rand(4, 16, 40, 40)
        loss = rand(7)

        with raises(ValueError):
            Output(x, loss)

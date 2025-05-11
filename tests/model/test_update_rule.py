"""
Ensures that the differentiable update rule is computed correctly.
"""
from numpy import float32 as npFloat32
from numpy.random import Generator
from pytest import fixture, mark
from torch import Tensor, equal, from_numpy, rand, tensor
from torch.nn import Conv2d, ReLU, Sequential
from torch.nn.init import constant_

from cellmorph.model import UpdateRule


class TestUpdateRule:
    """
    Test suite for :class:`UpdateRule`.
    """

    @mark.parametrize('state_channels, intermediate_channels', [
        (8, 64),
        (16, 128),
        (48, 384)
    ])
    def test_update_rule_constructs_layers_correctly_without_bias(
        self, state_channels: int, intermediate_channels: int
    ):
        expected = Sequential(
            Conv2d(in_channels=state_channels * 3,
                   out_channels=intermediate_channels, kernel_size=1,
                   padding=0),
            ReLU(),
            Conv2d(in_channels=intermediate_channels,
                   out_channels=state_channels, kernel_size=1, padding=0,
                   bias=False)
        )
        result = UpdateRule(state_channels, intermediate_channels)._layers

        constant_(expected[2].weight, 0.0)

        # First layer weights are randomly initialized (learned).
        assert isinstance(result[1], ReLU)
        assert equal(result[2].weight.data, expected[2].weight.data)
        assert result[2].bias is None

    @mark.parametrize('state_channels, intermediate_channels', [
        (8, 64),
        (16, 128),
        (48, 384)
    ])
    def test_update_rule_constructs_layers_correctly_with_bias(
        self, state_channels: int, intermediate_channels: int
    ):
        expected = Sequential(
            Conv2d(in_channels=state_channels * 3,
                   out_channels=intermediate_channels, kernel_size=1,
                   padding=0),
            ReLU(),
            Conv2d(in_channels=intermediate_channels,
                   out_channels=state_channels, kernel_size=1, padding=0,
                   bias=True)
        )
        result = UpdateRule(state_channels, intermediate_channels,
                            use_bias=True)._layers

        constant_(expected[2].weight, 0.0)
        constant_(expected[2].bias, 0.0)

        # First layer weights are randomly initialized (learned).
        assert isinstance(result[1], ReLU)
        assert equal(result[2].weight.data, expected[2].weight.data)
        assert equal(result[2].bias, expected[2].bias)

    @mark.parametrize('state_channels, intermediate_channels', [
        (16, 128),
    ])
    def test_update_rule_computes_correctly(
        self, state_channels: int, intermediate_channels: int, rng: Generator
    ):
        x_ = rand((1, state_channels * 3, 40, 40)).float()
        expected = Sequential(
            Conv2d(state_channels * 3,
                   out_channels=intermediate_channels, kernel_size=1,
                   padding=0),
            ReLU(),
            Conv2d(intermediate_channels,
                   out_channels=state_channels, kernel_size=1, padding=0,
                   bias=False)
        )
        result = UpdateRule(state_channels, intermediate_channels,
                              use_bias=False)

        constant_(expected[2].weight, 0.0)

        # Fix first layer weights.
        weights = rand(expected[0].weight.data.shape).float()

        expected[0].weight.data = weights
        result._layers[0].weight.data = weights

        assert equal(result(x_), expected(x_))

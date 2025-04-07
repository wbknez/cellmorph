"""
Ensures that individual automata perceive their environment (three by three
neighborhood) and evaluate it correctly.
"""
from numpy import cos as npCos, float32 as npFloat32, sin as npSin
from numpy.random import Generator
from pytest import fixture, mark
from torch import Tensor, equal, float32, from_numpy, outer, stack, tensor
from torch.nn.functional import conv2d

from cellmorph.model import PerceptionRule


@fixture(scope="function")
def x(state_channels: int, rng: Generator) -> Tensor:
    return from_numpy(rng.random((1, state_channels, 40, 40), dtype=npFloat32))


class TestPerceptionRule:
    """
    Test suite for :class:`PerceptionRule`.
    """

    @mark.parametrize('state_channels', [16, 25, 7])
    def test_sobel_weights_are_correct_by_example(self, state_channels: int):
        dx = tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float() / 8.0
        dy = tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).float() / 8.0
        i = tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).float()

        expected = stack([i, dx, dy]).repeat(state_channels, 1, 1).unsqueeze(1)
        result = PerceptionRule(state_channels, rotation=0,
                                normalize=False)._weights

        assert equal(result, expected)

    @mark.parametrize('state_channels', [16, 25, 7])
    def test_sobel_weights_are_correct_without_normalization_or_rotation(
        self, state_channels: int
    ):
        perceiver = PerceptionRule(state_channels, rotation=0, normalize=False)

        identify = outer(tensor([0, 1, 0]), tensor([0, 1, 0])).float()
        dx = outer(tensor([1, 2, 1]), tensor([-1, 0, 1])) / 8.0
        dy = dx.T.flipud()

        c, s = npCos(0), npSin(0)

        kernel = stack([identify, c * dx - s * dy, s * dx + c * dy])

        expected = kernel.repeat(state_channels, 1, 1).unsqueeze(1)
        result = perceiver._weights.data

        assert equal(result, expected)

    @mark.parametrize('state_channels, rotation',[
        (16, 0),
        (16, 90),
        (16, -36),
        (25, 0),
        (25, 32.5),
        (25, -97.22),
        (7, 0),
        (7, 15.37),
        (7, -180.2)
    ])
    def test_sobel_weights_are_correct_with_rotation_but_no_normalization(
        self, state_channels: int, rotation: float
    ):
        perceiver = PerceptionRule(state_channels, rotation=rotation,
                                   normalize=False)

        identify = outer(tensor([0, 1, 0]), tensor([0, 1, 0])).float()
        dx = outer(tensor([1, 2, 1]), tensor([-1, 0, 1])) / 8.0
        dy = dx.T.flipud()

        c, s = npCos(rotation), npSin(rotation)

        kernel = stack([identify, c * dx - s * dy, s * dx + c * dy])

        expected = kernel.repeat(state_channels, 1, 1).unsqueeze(1)
        result = perceiver._weights.data

        assert equal(result, expected)

    @mark.parametrize('state_channels, rotation',[
        (16, 0),
        (16, 90),
        (16, -36),
        (25, 0),
        (25, 32.5),
        (25, -97.22),
        (7, 0),
        (7, 15.37),
        (7, -180.2)
    ])
    def test_sobel_weights_are_correct_with_rotation_and_normalization(
        self, state_channels: int, rotation: float
    ):
        perceiver = PerceptionRule(state_channels, rotation=rotation,
                                   normalize=True)

        identify = outer(tensor([0, 1, 0]), tensor([0, 1, 0])).float()
        dx = outer(tensor([1, 2, 1]), tensor([-1, 0, 1])) / 8.0
        dy = dx.T.flipud()

        c, s = npCos(rotation), npSin(rotation)

        kernel = stack([identify, c * dx - s * dy, s * dx + c * dy])
        kernel = kernel.repeat(state_channels, 1, 1).unsqueeze(1)

        expected = kernel / state_channels
        result = perceiver._weights.data

        assert equal(result, expected)

    @mark.parametrize('state_channels, rotation',[
        (16, 0),
        (16, 90),
        (16, -36),
        (25, 0),
        (25, 32.5),
        (25, -97.22),
        (7, 0),
        (7, 15.37),
        (7, -180.2)
    ])
    def test_perceiver_computes_forward_pass_correctly(
        self, state_channels: int, rotation: float, x: Tensor
    ):
        perceiver = PerceptionRule(state_channels, rotation=rotation,
                                   normalize=False)

        identify = outer(tensor([0, 1, 0]), tensor([0, 1, 0])).float()
        dx = outer(tensor([1, 2, 1]), tensor([-1, 0, 1])) / 8.0
        dy = dx.T.flipud()

        c, s = npCos(rotation), npSin(rotation)

        kernel = stack([identify, c * dx - s * dy, s * dx + c * dy])
        kernel = kernel.repeat(state_channels, 1, 1).unsqueeze(1)

        expected = conv2d(x, weight=kernel, padding=1, groups=state_channels)
        result = perceiver(x)

        assert equal(result, expected)

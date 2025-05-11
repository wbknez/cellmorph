"""
Ensures that active automata detection works correctly.
"""
from numpy import float32
from numpy.random import Generator
from pytest import fixture, mark
from torch import Tensor, equal, tensor
from torch.nn.functional import max_pool2d

from cellmorph.model import is_active


@fixture(scope="function")
def x(rng: Generator) -> Tensor:
    return tensor(rng.random((1, 16, 40, 40), dtype=float32))


class TestIsActive:
    """
    Test suite for :meth:`is_active`.
    """

    @mark.parametrize('threshold', [0.1, 0.5, 0.9])
    def test_is_active_produces_correct_result(
        self, x: Tensor, threshold: float
    ):
        expected = max_pool2d(
            x.clone().detach()[:, 3, :, :].clip(0.0, 1.0),
            kernel_size=3, stride=1, padding=1
        )

        expected = expected > threshold
        result = is_active(x, threshold)

        assert equal(result, expected)

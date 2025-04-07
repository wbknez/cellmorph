"""
Ensures that image array creation from a model output tensor works correctly.
"""
from numpy import array_equal, float32, zeros
from numpy.random import Generator
from pytest import fixture
from torch import Tensor, from_numpy, tensor, uint8

from cellmorph.utils import to_rgb_array


@fixture(scope="function")
def x(rng: Generator) -> Tensor:
    return from_numpy(rng.random((1, 16, 40, 40), dtype=float32))


class TestToRGBArray:
    """
    Test suite for :meth:`to_rgb_array`.
    """

    def test_rgb_array_is_correct_without_premultiply(self, x: Tensor):
        rgb = x[:, :3, :, :].clip(min=0, max=1)
        alpha = x[:, 3:4, :, :].clip(min=0, max=1)

        expected = ((1.0 - alpha + rgb) * 255.0).to(uint8)
        result = to_rgb_array(x)

        print(result.shape)

        assert array_equal(result, expected.numpy())

    def test_rgb_array_is_correct_with_premultiply(self, x: Tensor):
        rgb = x[:, :3, :, :].clip(min=0, max=1)
        alpha = x[:, 3:4, :, :].clip(min=0, max=1)

        rgb *= alpha

        expected = ((1.0 - alpha + rgb) * 255.0).to(uint8)
        result = to_rgb_array(x, premultiply=True)

        assert array_equal(result, expected.numpy())

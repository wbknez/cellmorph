"""
Ensures that converting a tensor to an image is correct.
"""
from numpy import array, array_equal, float32, uint8
from numpy.random import Generator
from PIL import Image
from pytest import mark, raises
from torch import Tensor, from_numpy

from cellmorph.image import from_tensor


class TestFromTensor:
    """
    Test suite for :meth:`from_tensor`.
    """

    @mark.parametrize('sample_count, state_channels, height, width', [
        (1, 16, 60, 60)
    ])
    def test_from_tensor_raises_if_x_dims_is_greater_than_4( self, x: Tensor):
        with raises(ValueError):
            from_tensor(x.unsqueeze(0))

    @mark.parametrize('sample_count, state_channels, height, width', [
        (1, 16, 60, 60)
    ])
    def test_from_tensor_raises_if_x_dims_is_less_than_3( self, x: Tensor):
        with raises(ValueError):
            from_tensor(x.flatten())

        with raises(ValueError):
            rows = x.shape[0] * x.shape[1]
            cols = x.shape[2] * x.shape[3]

            from_tensor(x.reshape(rows, cols))

    @mark.parametrize('sample_count, state_channels, height, width', [
        (1, 16, 40, 40),
        (1, 4, 30, 15),
        (1, 3, 97, 42),
    ])
    def test_from_tensor_converts_tensor_correctly(self, x: Tensor):
        y = x.clone().detach()

        if len(y.shape) == 4:
            y = y.squeeze(0)

        if y.shape[0] > 4:
            y = y[:4, ...]

        arr = y.clip(min=0.0, max=1.0).permute(1, 2, 0).numpy()
        arr = (arr * 255).astype(uint8)
        mode = "RGBA" if arr.shape[-1] >= 4 else "RGB"

        expected = Image.fromarray(arr, mode=mode)
        result = from_tensor(x)

        assert result.size == expected.size
        assert result.mode == expected.mode
        assert array_equal(array(result), array(expected))
        assert result == expected

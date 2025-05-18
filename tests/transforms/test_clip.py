"""
Ensures that constraining tensor values to an interval performs as expected.
"""
from PIL.Image import Image
from pytest import mark, raises
from torch import Tensor, equal, rand, randint, zeros_like

from cellmorph.transforms.tensor import Clip


class TestClip:
    """"
    Test suite for :class:`Clip`.
    """

    @mark.parametrize('min', [ -0.0001, 1.0001 ])
    def test_constructor_raises_if_min_is_out_of_bounds(self, min: float):
        with raises(ValueError):
            Clip(min, 0.5)

    def test_constructor_raises_if_max_is_less_than_min(self):
        with raises(ValueError):
            Clip(0.5, 0.3)

    @mark.parametrize('width, height, color_channels', [
        (72, 72, 4)
    ])
    def test_transform_raises_if_inpt_is_not_a_tensor(self, img: Image):
        with raises(TypeError):
            Clip(0.0, 1.0).transform(img)

    @mark.parametrize('min, max, tnsr', [
        (0.0, 1.0, rand((100,))),
        (0.2, 0.5, rand(15, 15)),
        (0.43, 0.95, rand((8, 16, 23, 25)))
    ])
    def test_transform_constrains_values_correctly(
        self, min: float, max: float, tnsr: Tensor
    ):
        flat = tnsr.flatten()
        expected = zeros_like(flat)

        for i in range(tnsr.numel()):
            value = flat[i]

            if value < min:
                expected[i] = min
            elif value > max:
                expected[i] = max
            else:
                expected[i] = value

        expected = expected.reshape(tnsr.shape)
        result = Clip(min, max).transform(tnsr)

        assert equal(result, expected)

"""
Ensures that model seed creation works as expected.
"""
from numpy import float32, zeros
from pytest import mark, raises
from torch import equal, tensor

from cellmorph.utils import create_model_seed


class TestCreateModelSeed:
    """
    Test suite for :meth:`create_model_seed`.
    """

    @mark.parametrize('width', [0, -1])
    def test_model_seed_raises_if_width_is_not_positive(self, width: int):
        with raises(ValueError):
            create_model_seed(width, 1, 1)

    @mark.parametrize('height', [0, -1])
    def test_model_seed_raises_if_height_is_not_positive(self, height: int):
        with raises(ValueError):
            create_model_seed(1, height, 1)

    @mark.parametrize('channels', [0, -1])
    def test_model_seed_raises_if_channels_is_not_positive(self, channels: int):
        with raises(ValueError):
            create_model_seed(1, 1, channels)

    @mark.parametrize('width, height', [(40, 40), (10, 50), (75, 15)])
    def test_model_seed_only_has_one_active_channel(
        self, width: int, height: int
    ):
        expected = zeros((16, height, width), dtype=float32)
        result = create_model_seed(width, height, 16)

        expected[3:, height // 2, width // 2] = 1.0

        assert equal(result, tensor(expected))

    @mark.parametrize('width, height', [(40, 40), (10, 50), (75, 15)])
    def test_model_seed_shape_is_correct(self, width: int, height: int):
        seed = create_model_seed(width, height, 16)

        assert seed.shape == (16, height, width)

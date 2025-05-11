"""
Ensures that damage mask creation works as expected.
"""
from pytest import mark
from torch import equal, linspace, manual_seed, rand
from torch.distributions.uniform import Uniform

from cellmorph.data import damage_mask
from cellmorph.image import Dimension


class TestDamageMask:
    """
    Test suite for :meth:`damage_mask`.
    """

    @mark.parametrize('sample_count, size, seed', [
        (10, Dimension(32, 24), 29011823),
        (32, Dimension(72, 72), 91988395),
        (3, Dimension(16, 24), 34902948),
    ])
    def test_damage_mask_is_computed_correctly(
        self, sample_count: int, size: Dimension, seed: int
    ):
        manual_seed(seed)

        theta = linspace(-1.0, 1.0, size.width)[None, None, :]
        phi = linspace(-1.0, 1.0, size.height)[None, :, None]

        center = Uniform(-0.5, 0.5).sample((2, sample_count, 1, 1))
        r = Uniform(0.1, 0.4).sample((sample_count, 1, 1))

        x = (theta - center[0]) / r
        y = (phi - center[1]) / r

        mask = ((x * x + y * y) < 1.0).float().unsqueeze(1)

        manual_seed(seed)

        expected = 1.0 - mask
        result = damage_mask(sample_count, size)

        assert equal(result, expected)

    @mark.parametrize('sample_count, size', [
        (10, Dimension(32, 24)),
        (32, Dimension(72, 72)),
        (3, Dimension(16, 24)),
    ])
    def test_damage_mask_result_can_be_applied_correctly(
        self, sample_count: int, size: Dimension
    ):
        mask = damage_mask(sample_count, size)
        data = rand((sample_count, 16, size.height, size.width))

        expected = data.clone().detach()

        for x in range(data.shape[0]):
            expected[x, ...] *= mask[x, ...]

        result = data * mask

        assert equal(result, expected)

"""
Ensures that model optimization configuration initialization works correctly.
"""
from pytest import mark, raises

from cellmorph.config import OptimizerConfiguration


class TestOptimizerConfiguration:
    """
    Test suite for :class:`OptimizerConfiguration`.
    """

    @mark.parametrize('milestones', [ None, [] ])
    def test_optimizer_configuration_raises_if_milestones_are_not_valid(
        self, milestones: list[int]
    ):
        with raises(ValueError):
            OptimizerConfiguration(milestones, 2e-3, 0.01, 20)

    @mark.parametrize('milestones', [ [1, 0], [2, 3, -1] ])
    def test_optimizer_configuration_raises_if_milestones_are_not_positive(
        self, milestones: list[int]
    ):
        with raises(ValueError):
            OptimizerConfiguration(milestones, 2e-3, 0.01, 20)

    @mark.parametrize('learning_rate', [ 0, -0.0001 ])
    def test_optimizer_configuration_raises_if_learning_rate_is_negative(
        self, learning_rate: float
    ):
        with raises(ValueError):
            OptimizerConfiguration([2000], learning_rate, 0.01, 20)

    @mark.parametrize('gamma', [ 0.0, -0.0001 ])
    def test_optimizer_configuration_raises_if_gamma_is_negative(
        self, gamma: float
    ):
        with raises(ValueError):
            OptimizerConfiguration([2000], 2e-3, gamma, 20)

    @mark.parametrize('gradient_cutoff', [ 0, -1 ])
    def test_optimizer_configuration_raises_if_gradient_cutoff_is_negative(
        self, gradient_cutoff: int
    ):
        with raises(ValueError):
            OptimizerConfiguration([2000], 2e-3, 0.01, gradient_cutoff)

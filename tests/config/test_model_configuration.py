"""
Ensures that model configuration initialization works correctly.
"""
from numpy import pi
from pytest import mark, raises

from cellmorph.config import ModelConfiguration


class TestModelConfiguration:
    """
    Test suite for :class:`ModelConfiguration`.
    """

    @mark.parametrize('state_channels', [ 0, -1 ])
    def test_model_configuration_raises_if_state_channels_are_not_positive(
        self, state_channels: int
    ):
        with raises(ValueError):
            ModelConfiguration(state_channels, 1, True, 0.0, 1, 0.1, 0.5,
                               True)

    @mark.parametrize('intermediate_channels', [ 0, -1 ])
    def test_model_configuration_intermediate_channels_are_not_positive(
        self, intermediate_channels: int
    ):
        with raises(ValueError):
            ModelConfiguration(1, intermediate_channels, True, 0.0, 1, 0.1,
                               0.5, True)

    @mark.parametrize('rotation', [-0.0001, 2 * pi])
    def test_model_configuration_raises_if_rotation_is_out_of_bounds(
        self, rotation: float
    ):
        with raises(ValueError):
            ModelConfiguration(1, 1, True, rotation, 1, 0.1, 0.5, True)

    @mark.parametrize('step_size', [ 0.0, -1.0 ])
    def test_model_configuration_raises_if_step_size_is_not_positive(
        self, step_size: float
    ):
        with raises(ValueError):
            ModelConfiguration(1, 1, True, 0.0, step_size, 0.1, 0.5, True)

    @mark.parametrize('threshold', [ 0.0, 1.0001 ])
    def test_model_configuration_raises_if_threshold_is_out_of_bounds(
        self, threshold: float
    ):
        with raises(ValueError):
            ModelConfiguration(1, 1, True, 0.0, 0.5, threshold, 0.5, True)

    @mark.parametrize('update_rate', [ 0.0, 1.0001 ])
    def test_model_configuration_raises_if_update_rate_is_out_of_bounds(
        self, update_rate: float
    ):
        with raises(ValueError):
            ModelConfiguration(1, 1, True, 0.0, 0.5, 0.1, update_rate, True)

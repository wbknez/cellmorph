"""
Ensures that training configuration initialization works correctly. 
"""
from pytest import mark, raises

from cellmorph.config import Interval, TrainingConfiguration


class TestTrainingConfiguration:
    """
    Test suite for :class:`TrainingConfiguration`.
    """

    @mark.parametrize('strategy', [ None, "" ])
    def test_train_config_raises_if_strategy_is_not_valid(self, strategy: str):
        with raises(ValueError):
            TrainingConfiguration(strategy, 8, Interval(1, 2), 8000, 1, 3)

    @mark.parametrize('batch_size', [ 0, -1 ])
    def test_train_config_raises_if_batch_size_is_not_positive(
        self, batch_size: int
    ):
        with raises(ValueError):
            TrainingConfiguration("growth", batch_size, Interval(1, 2), 8000, 1,
                                  3)

    def test_train_config_raises_if_interval_is_not_valid(self):
        with raises(ValueError):
            TrainingConfiguration("growth", 8, None, 8000, 1, 3)

    @mark.parametrize('epochs', [ 0, -1 ])
    def test_train_config_raises_if_epochs_is_not_positive(self, epochs: int):
        with raises(ValueError):
            TrainingConfiguration("growth", 8, Interval(1, 2), epochs, 1, 3)

    @mark.parametrize('reset_count', [ -1, -2 ])
    def test_train_config_raises_is_reset_count_is_negative(
        self, reset_count: int
    ):
        with raises(ValueError):
            TrainingConfiguration("growth", 8, Interval(1, 2), 8000,
                                  reset_count, 3)

    @mark.parametrize('damage_count', [ -1, -2 ])
    def test_train_config_raises_is_reset_count_is_negative(
        self, damage_count: int
    ):
        with raises(ValueError):
            TrainingConfiguration("growth", 8, Interval(1, 2), 8000,
                                  1, damage_count)

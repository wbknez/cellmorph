"""
Ensures that counting trainable parameters is correct.
"""
from pytest import mark
from torch import Tensor
from torch.nn import Conv2d, Module

from cellmorph.utils import count_parameters


class ExampleModel(Module):
    
    def __init__(self, in_ch: int, hd_ch0: int, hd_ch1: int, ot_ch: int):
        super().__init__()

        self._conv0 = Conv2d(in_ch, hd_ch0, 1, padding=0)
        self._conv1 = Conv2d(hd_ch0, hd_ch1, 1, padding=0)
        self._conv2 = Conv2d(hd_ch1, ot_ch, 1, padding=0)

    def forward(self, x: Tensor):
        y = self._conv0(x)
        y = self._conv1(y)

        return self._conv2(y)


class TestCountParameters:
    """
    Test suite for :meth:`count_parameters`.
    """

    @mark.parametrize('in_ch, hd_ch0, hd_ch1, ot_ch', [
        (16, 512, 128, 48),
        (48, 96, 23, 2),
        (97, 1024, 516, 38)
    ])
    def test_count_parameters_counts_parameters_correctly(
        self, in_ch: int, hd_ch0: int, hd_ch1: int, ot_ch: int
    ):
        model = ExampleModel(in_ch, hd_ch0, hd_ch1, ot_ch)

        expected = 0

        for param in model.parameters():
            product = 1

            for dim in param.size():
                product *= dim

            expected += product

        result = count_parameters(model)

        assert result == expected

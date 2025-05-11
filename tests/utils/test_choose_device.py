"""
Ensures that custom device selection works as expected.
"""
from pytest import mark
from torch import device
from torch.cuda import is_available as is_cuda_available
from torch.backends.mps import is_available as is_mps_available

from cellmorph.utils import choose_device


class TestChooseDevice:
    """
    Test suite for :meth:`choose_device`.
    """

    def test_choose_device_chooses_best_device_on_platform(self):
        expected = "cpu"

        if is_cuda_available():
            expected = "cuda"
        elif is_mps_available():
            expected = "mps"

        result = choose_device()
        
        assert result == device(expected)
    
    @mark.parametrize('choice', [ "cuda", "mps", "cpu" ])
    def test_choose_device_respects_override(self, choice: str):
        expected = device(choice)
        result = choose_device(choice)

        assert result == expected

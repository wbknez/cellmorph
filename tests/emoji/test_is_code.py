"""
Ensures that hexadecimal detection works as expected.
"""
from pytest import mark

from cellmorph.emoji import is_code


class TestIsCode:
    """
    Test suite for :meth:`is_code`.
    """

    @mark.parametrize('code', [ "not_a_code", "019294eoi" ])
    def test_is_code_returns_false_if_not_hex(self, code: str):
        assert not is_code(code)

    @mark.parametrize('code', [ "0x012023", "0x01af6" ])
    def test_is_code_returns_true_if_hex(self, code: str):
        assert is_code(code)

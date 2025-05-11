"""
Ensures that creating a unique folder name for model output works as expected.
"""
from datetime import datetime

from cellmorph.utils import unique_name


class TestUniqueName:
    """
    Test suite for :meth:`unique_name`.
    """

    def test_unique_name_produces_formatted_name(self):
        format = "{}-%d%m%Y-%H%M"
        name = "some_model_name"

        ts = datetime.now()

        expected = ts.strftime(format.format(name))
        result = unique_name(name, ts)

        assert result == expected

    def test_unique_name_produces_formatted_name_with_custom_format(self):
        format = "{}-%H%M_%Y%d"
        name = "some_model_name"

        ts = datetime.now()

        expected = ts.strftime(format.format(name))
        result = unique_name(name, ts, format)

        assert result == expected

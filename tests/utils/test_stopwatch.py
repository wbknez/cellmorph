"""
Ensures that stopwatch produces correctly formatted elapsed time.
"""
from datetime import datetime

from cellmorph.utils import Stopwatch


class TestStopwatch:
    """
    Test suite for :class:Stopwatch.
    """

    def test_elapsed_time_is_formatted_correctly(self):
        summation = 0
        timer = Stopwatch()

        timer._start = datetime(2025, 3, 20, 10, 30, 11)
        timer._end = datetime(2025, 3, 20, 11, 9, 56)

        elapsed = timer._end - timer._start

        hours = elapsed.seconds // 3600
        minutes = (elapsed.seconds % 3600) // 60
        seconds = elapsed.seconds % 60

        nouns = (
            "hours" if hours != 1 else "hour",
            "minutes" if minutes != 1 else "minute",
            "seconds" if seconds != 1 else "second"
        )
        format = "{} {}, {} {}, and {} {}"

        expected = format.format(
            hours, nouns[0], minutes, nouns[1], seconds, nouns[2]
        )
        result = timer.elapsed()

        assert result == expected

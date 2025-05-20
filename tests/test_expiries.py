import datetime as dt
import unittest
from unittest.mock import patch

import option_analysis as oa

class ExpiryWithinTests(unittest.TestCase):
    def test_expiries_within_sorted(self):
        today = dt.date.today()
        dates = [
            today + dt.timedelta(days=d) for d in [1, 15, 7]
        ]
        options = [d.strftime("%Y-%m-%d") for d in dates]

        class FakeTicker:
            def __init__(self, opts):
                self.options = opts

        with patch("option_analysis.yf.Ticker", return_value=FakeTicker(options)):
            res = oa.expiries_within(max_days=14)

        expected = sorted(
            (today + dt.timedelta(days=d)).strftime("%Y-%m-%d")
            for d in [1, 7]
        )
        self.assertEqual(res, expected)

if __name__ == "__main__":
    unittest.main()

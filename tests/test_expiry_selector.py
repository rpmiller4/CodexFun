import datetime as dt
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import expiry_selector


class ExpirySelectorTests(unittest.TestCase):
    def test_expiries_within_sorted(self):
        class FakeTicker:
            def __init__(self, opts):
                self.options = opts

        mock_opts = [
            "2024-01-10",
            "2024-01-05",
            "2023-12-31",
            "2024-01-20",
        ]
        tkr = FakeTicker(mock_opts)

        class FakeDate(dt.date):
            @classmethod
            def today(cls):
                return dt.date(2024, 1, 1)

        with patch.object(expiry_selector, "dt", SimpleNamespace(date=FakeDate, datetime=dt.datetime)):
            res = expiry_selector.expiries_within(tkr, max_days=14)
        self.assertEqual(res, ["2024-01-05", "2024-01-10"])


if __name__ == "__main__":
    unittest.main()

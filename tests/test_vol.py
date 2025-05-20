import math
import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vol_utils import resolve_iv, iv_is_valid, IVSource


class VolUtilsTests(unittest.TestCase):
    def test_resolve_iv_atm(self):
        chain = pd.DataFrame({"strike": [100, 105], "impliedVolatility": [0.2, 0.21]})
        iv, src = resolve_iv(0.0, chain, spot=102, days=10)
        self.assertEqual(src, IVSource.ATM)
        self.assertAlmostEqual(iv, 0.2, places=3)

    def test_resolve_iv_vix(self):
        chain = pd.DataFrame({"strike": [], "impliedVolatility": []})
        with patch("vol_utils.yf.Ticker") as FakeTicker:
            FakeTicker.return_value.history.return_value = pd.DataFrame({"Close": [20.0]})
            iv, src = resolve_iv(0.0, chain, spot=100, days=15)
        expected = 0.20 * math.sqrt(15 / 30.0)
        self.assertEqual(src, IVSource.VIX)
        self.assertAlmostEqual(iv, expected, places=6)

    def test_cli_help_heading(self):
        import subprocess
        result = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "..", "cli.py"), "--help"], capture_output=True, text=True)
        self.assertIn("~", result.stdout)
        self.assertNotIn("\u2248", result.stdout)

    def test_iv_is_valid_threshold(self):
        self.assertTrue(iv_is_valid(0.1, 0.05))
        self.assertFalse(iv_is_valid(0.04, 0.05))


if __name__ == "__main__":
    unittest.main()

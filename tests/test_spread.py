import datetime as dt
import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import OptionContract
from vol_utils import IVSource
import spread_analysis as sa


class SpreadTests(unittest.TestCase):
    def setUp(self):
        self.today = dt.date.today()
        self.expiry = self.today + dt.timedelta(days=10)
        self.underlying = 100.0
        self.iv = 0.25

    def _make_option(self, strike: float, price: float) -> OptionContract:
        return OptionContract(
            strike=strike,
            expiry=self.expiry,
            option_type="put",
            last_price=price,
            iv=self.iv,
            iv_src=IVSource.ORIG,
            underlying_price=self.underlying,
        )

    def test_spread_values(self):
        short = self._make_option(100, 3.0)
        long = self._make_option(95, 1.0)
        spread = sa.make_spread(sa.SpreadType.BULL_PUT, short, long, today=self.today)
        self.assertAlmostEqual(spread.credit, 2.0)
        self.assertAlmostEqual(spread.width, 5.0)
        self.assertAlmostEqual(spread.credit_pct, 40.0)

    def test_pop_decreases_with_strike(self):
        short1 = self._make_option(100, 3.0)
        long1 = self._make_option(95, 1.0)
        spread1 = sa.make_spread(sa.SpreadType.BULL_PUT, short1, long1, today=self.today)

        short2 = self._make_option(105, 3.0)
        long2 = self._make_option(100, 1.0)
        spread2 = sa.make_spread(sa.SpreadType.BULL_PUT, short2, long2, today=self.today)

        self.assertGreater(spread1.pop, spread2.pop)

    def test_filter_credit_spreads(self):
        short = self._make_option(100, 3.0)
        long = self._make_option(95, 1.0)
        s1 = sa.make_spread(sa.SpreadType.BULL_PUT, short, long, today=self.today)
        s2 = sa.make_spread(sa.SpreadType.BULL_PUT, short, self._make_option(99, 2.8), today=self.today)
        spreads = sa.filter_credit_spreads([s1, s2], pop_min=0.0, credit_min_pct=30.0)
        self.assertEqual(spreads[0], s1)
        self.assertEqual(len(spreads), 1)

    def test_get_credit_spreads_skips_bad_iv(self):
        import pandas as pd
        from unittest.mock import patch

        # fake option chain with zero IVs so fallback also fails
        df = pd.DataFrame(
            {
                "strike": [100, 95],
                "lastPrice": [2.0, 0.5],
                "impliedVolatility": [0.0, 0.0],
            }
        )

        class FakeChain:
            def __init__(self, frame):
                self.puts = frame
                self.calls = frame

        class FakeTicker:
            def history(self, period="1d"):
                return pd.DataFrame({"Close": [100.0]})

            def option_chain(self, expiry):
                return FakeChain(df)

        with patch("yfinance.Ticker", return_value=FakeTicker()), patch(
            "vol_utils.vix_sigma",
            return_value=0.0,
        ):
            spreads = sa.get_credit_spreads(
                "2099-01-01", width=5.0, spread_type=sa.SpreadType.BULL_PUT
            )
        self.assertEqual(len(spreads), 0)

    def test_get_credit_spreads_iv_sources_and_widths(self):
        import pandas as pd
        from unittest.mock import patch

        df = pd.DataFrame(
            {
                "strike": [100, 98, 95, 90],
                "lastPrice": [1.5, 0.6, 1.0, 0.5],
                "impliedVolatility": [0.0, 0.16, 0.0, 0.0],
            }
        )

        class FakeChain:
            def __init__(self, frame):
                self.puts = frame
                self.calls = frame

        class FakeTicker:
            def history(self, period="1d"):
                return pd.DataFrame({"Close": [100.0]})

            def option_chain(self, expiry):
                return FakeChain(df)

        with patch("yfinance.Ticker", return_value=FakeTicker()), patch(
            "vol_utils.vix_sigma",
            return_value=0.2,
        ):
            w2 = sa.get_credit_spreads(
                "2099-01-01", width=2.0, spread_type=sa.SpreadType.BULL_PUT
            )
            w5 = sa.get_credit_spreads(
                "2099-01-01", width=5.0, spread_type=sa.SpreadType.BULL_PUT
            )

        self.assertEqual(len(w2), 1)
        self.assertGreaterEqual(len(w5), 1)

        s2 = w2[0]
        s5 = next(s for s in w5 if s.short.strike == 95)

        self.assertEqual(s2.width, 2.0)
        self.assertEqual(s5.width, 5.0)
        self.assertNotEqual(s2.credit_pct, s5.credit_pct)
        self.assertNotEqual(s2.pop, s5.pop)
        self.assertEqual(s2.iv_short_src, sa.IVSource.VIX)
        self.assertEqual(s2.iv_long_src, sa.IVSource.ORIG)
        self.assertEqual(s5.iv_short_src, sa.IVSource.VIX)
        self.assertEqual(s5.iv_long_src, sa.IVSource.VIX)

    def test_filter_sorted_by_ev(self):
        short_a = self._make_option(99, 2.0)
        long_a = self._make_option(94, 0.5)
        s_a = sa.make_spread(sa.SpreadType.BULL_PUT, short_a, long_a, today=self.today)

        short_b = self._make_option(101, 4.0)
        long_b = self._make_option(96, 0.8)
        s_b = sa.make_spread(sa.SpreadType.BULL_PUT, short_b, long_b, today=self.today)

        spreads = sa.filter_credit_spreads([s_a, s_b], pop_min=0.0, credit_min_pct=0.0)
        self.assertEqual(spreads[0], s_b)


if __name__ == "__main__":
    unittest.main()

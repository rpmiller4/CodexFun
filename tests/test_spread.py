import datetime as dt
import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import OptionContract
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


if __name__ == "__main__":
    unittest.main()

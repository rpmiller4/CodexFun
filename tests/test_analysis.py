import datetime as dt
import unittest

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import option_analysis as oa
from vol_utils import IVSource


class OptionAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.today = dt.date.today()
        self.expiry = self.today + dt.timedelta(days=21)
        self.underlying = 430.0
        self.iv = 0.25

    def _make_option(self, strike: float) -> oa.OptionContract:
        T = 21 / 365
        price = oa.black_scholes_price(
            self.underlying, strike, T, 0.05, self.iv, option_type="call"
        )
        return oa.OptionContract(
            strike=strike,
            last_price=price,
            iv=self.iv,
            iv_src=IVSource.ORIG,
            expiry=self.expiry,
            option_type="call",
            underlying_price=self.underlying,
        )

    def test_pop_decreases_with_strike(self):
        opt_low = self._make_option(430)
        opt_high = self._make_option(440)
        stats_low = oa.compute_option_metrics(opt_low, today=self.today)
        stats_high = oa.compute_option_metrics(opt_high, today=self.today)
        self.assertGreater(stats_low.pop, stats_high.pop)

    def test_intrinsic_value(self):
        opt = self._make_option(435)
        stats = oa.compute_option_metrics(opt, today=self.today)
        self.assertEqual(stats.intrinsic_value, 0.0)

    def test_time_value_decay(self):
        opt = self._make_option(435)
        prices = oa.simulate_value_decay(opt, [21, 10, 1])
        intrinsic = max(self.underlying - 435, 0)
        time_values = [p - intrinsic for p in prices]
        self.assertGreater(time_values[0], time_values[1])
        self.assertGreater(time_values[1], time_values[2])

    def test_days_to_expiry(self):
        opt = self._make_option(435)
        stats = oa.compute_option_metrics(opt, today=self.today)
        self.assertEqual(stats.days_to_expiry, 21)

    def test_filter_options_by_pop_and_timevalue_percent(self):
        opt1 = self._make_option(420)
        opt2 = self._make_option(435)
        stats = [
            oa.compute_option_metrics(opt1, today=self.today),
            oa.compute_option_metrics(opt2, today=self.today),
        ]
        filtered = oa.filter_options_by_pop_and_timevalue_percent(
            stats, pop_threshold=0.65, tv_threshold=0.25
        )
        self.assertEqual(len(filtered), 1)
        self.assertAlmostEqual(filtered[0].strike, 420)


if __name__ == "__main__":
    unittest.main()


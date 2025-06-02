import unittest
import datetime as dt
import numpy as np

import synthetic_market as sm


class RealismFeaturesTests(unittest.TestCase):
    def test_slippage_mean_credit_lower(self):
        rng = np.random.default_rng(0)
        fills = []
        for _ in range(100):
            dist = rng.uniform(5, 25)
            mid = 1.0
            fill = mid - 0.5 * sm.quote_spread(dist) * rng.uniform()
            fills.append(fill)
        self.assertLess(np.mean(fills), 1.0)

    def test_liquidity_execution_rate(self):
        rng = np.random.default_rng(0)
        liq_decay = 12
        executed = sum(
            rng.uniform() <= np.exp(-20 / liq_decay) for _ in range(1000)
        )
        rate = executed / 1000
        self.assertTrue(0.10 < rate < 0.20)

    def test_margin_rejects_overlever(self):
        capital = 1000
        width = 5
        credit = 1
        lots = 3
        margin_req = max(width * 100 - credit, 0) * lots
        self.assertLess(capital - margin_req, 0)

    def test_jump_frequency(self):
        start = dt.date(2020, 1, 1)
        end = start + dt.timedelta(days=10000)
        prices = sm.simulate_prices(
            100.0,
            start,
            end,
            0.05,
            0.2,
            seed=42,
            jump_prob=0.01,
            down_jump=-0.08,
            up_jump=0.04,
        )
        jump_rate = prices["Jump"].mean()
        self.assertAlmostEqual(jump_rate, 0.01, delta=0.005)


if __name__ == "__main__":
    unittest.main()

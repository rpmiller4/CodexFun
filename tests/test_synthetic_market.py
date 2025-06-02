import unittest
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import synthetic_market as sm
import datetime as dt

class SyntheticMarketTests(unittest.TestCase):
    def test_price_determinism(self):
        cfg = {
            "start_price": 100.0,
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "mu": 0.05,
            "sigma": 0.2,
            "seed": 42,
        }
        p1 = sm.simulate_prices(100.0, dt.date(2023,1,1), dt.date(2023,12,31), 0.05, 0.2, seed=42)
        p2 = sm.simulate_prices(100.0, dt.date(2023,1,1), dt.date(2023,12,31), 0.05, 0.2, seed=42)
        self.assertTrue((p1['Close'] == p2['Close']).all())

    def test_run_simulation(self):
        cfg = {
            "start_price": 100.0,
            "start_date": "2022-01-01",
            "end_date": "2023-12-31",
            "mu": 0.05,
            "sigma": 0.2,
            "seed": 1,
        }
        res = sm.run_simulation(cfg)
        self.assertGreater(len(res.trades), 0)
        self.assertIn('Equity', res.equity_curve.columns)
        # deterministic result for given seed
        res2 = sm.run_simulation(cfg)
        self.assertAlmostEqual(res.total_return, res2.total_return)

if __name__ == '__main__':
    unittest.main()

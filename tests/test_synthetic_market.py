import unittest
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import synthetic_market as sm
import numpy as np
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

    def test_credit_ratio_function(self):
        self.assertAlmostEqual(sm.credit_ratio(0, 1), 0.45)
        self.assertAlmostEqual(sm.credit_ratio(10, 1), 0.15, places=3)

    def test_commissions_reduce_equity(self):
        base_cfg = {
            "start_price": 100.0,
            "start_date": "2023-01-02",
            "end_date": "2023-03-01",
            "mu": 0.05,
            "sigma": 0.2,
            "seed": 2,
        }
        no_comm = sm.run_simulation(base_cfg)
        with_comm = sm.run_simulation({**base_cfg, "commissions": True})
        self.assertLess(with_comm.equity_curve["Equity"].iloc[-1], no_comm.equity_curve["Equity"].iloc[-1])
        expected = sum((1.5 + 0.65 * (t.lots * 2)) * 2 for t in with_comm.trades)
        diff = no_comm.equity_curve["Equity"].iloc[-1] - with_comm.equity_curve["Equity"].iloc[-1]
        self.assertAlmostEqual(diff, expected, places=2)

    def test_heston_path_properties(self):
        cfg = {
            "start_price": 100.0,
            "start_date": "2023-01-01",
            "end_date": "2023-03-01",
            "mu": 0.05,
            "sigma": 0.2,
            "seed": 0,
            "use_heston": True,
            "heston": sm.DEFAULT_CONFIG["heston"],
        }
        prices = sm.simulate_heston_prices(cfg)
        n_days = (dt.datetime.strptime(cfg["end_date"], "%Y-%m-%d").date() - dt.datetime.strptime(cfg["start_date"], "%Y-%m-%d").date()).days + 1
        self.assertEqual(len(prices), n_days)
        self.assertTrue((prices["Var"] >= 0).all())
        self.assertGreater(np.std(np.diff(prices["Var"])), 0)

if __name__ == '__main__':
    unittest.main()

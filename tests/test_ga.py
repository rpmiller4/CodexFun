import synthetic_market as sm


def test_ga_fitness_deterministic():
    cfg = {
        "start_price": 100.0,
        "start_date": "2023-01-01",
        "end_date": "2023-06-30",
        "mu": 0.05,
        "sigma": 0.2,
        "ga_ensemble_n": 3,
    }
    f1 = sm.ga_fitness(cfg)
    f2 = sm.ga_fitness(cfg)
    assert f1 == f2

    f3 = sm.ga_fitness({**cfg, "mu": 0.06})
    assert f1 != f3

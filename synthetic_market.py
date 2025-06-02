from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd


@dataclass
class Trade:
    """Represents a single credit spread trade."""

    open_date: dt.date
    expiry_date: dt.date
    trade_type: str  # "bull_put" or "bear_call"
    short_strike: float
    long_strike: float
    width: float
    credit: float
    lots: int
    result: float | None = None  # P&L realized at expiry


@dataclass
class SimulationResult:
    equity_curve: pd.DataFrame
    trades: List[Trade]
    total_return: float
    win_rate: float
    max_drawdown: float


DEFAULT_CONFIG: Dict[str, object] = {
    "start_price": 400.0,
    "start_date": "2022-01-01",
    "end_date": "2025-12-31",
    "mu": 0.07,
    "sigma": 0.20,
    "vol_model": "gbm",  # "gbm" or "heston"
    "kappa": 1.5,
    "theta": 0.04,
    "xi": 0.3,
    "rho": -0.7,
    "v0": 0.04,
    "ema_period": 50,
    "risk_fraction": 0.03,
    "max_lots": 6,
    "short_leg_distance": (10, 15),
    "spread_width_range": (1, 8),
    "trade_credit_ratio": 0.30,
    "option_spacing": 1,
    "expiry_days": 10,
    "seed": 123,
}


def simulate_prices(
    start_price: float,
    start_date: dt.date,
    end_date: dt.date,
    mu: float,
    sigma: float,
    seed: int = 0,
) -> pd.DataFrame:
    """Return DataFrame of simulated prices using geometric Brownian motion."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")
    n = len(dates)
    dt_frac = 1 / 252.0
    prices = np.empty(n)
    prices[0] = start_price
    for i in range(1, n):
        z = rng.standard_normal()
        prices[i] = prices[i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt_frac + sigma * np.sqrt(dt_frac) * z)
    return pd.DataFrame({"Date": dates.date, "Close": prices})


def heston_path(
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    s0: float,
    dt_frac: float,
    steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return price and variance paths using Euler-Milstein discretisation."""
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(steps)
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(steps)
    v = np.empty(steps + 1)
    s = np.empty(steps + 1)
    v[0] = v0
    s[0] = s0
    for t in range(steps):
        v[t + 1] = np.abs(v[t] + kappa * (theta - v[t]) * dt_frac + xi * np.sqrt(v[t] * dt_frac) * z2[t])
        s[t + 1] = s[t] * np.exp((r - 0.5 * v[t]) * dt_frac + np.sqrt(v[t] * dt_frac) * z1[t])
    return s, v


def simulate_prices_heston(
    start_price: float,
    start_date: dt.date,
    end_date: dt.date,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    seed: int = 0,
) -> pd.DataFrame:
    """Return DataFrame of prices simulated via the Heston model."""
    dates = pd.date_range(start_date, end_date, freq="D")
    steps = len(dates) - 1
    dt_frac = 1 / 252.0
    # use risk-neutral drift equal to the mean return
    s, _ = heston_path(mu, kappa, theta, xi, rho, v0, start_price, dt_frac, steps, seed)
    return pd.DataFrame({"Date": dates.date, "Close": s})


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Return exponential moving average for ``series``."""
    return series.ewm(span=period, adjust=False).mean()


def run_simulation(config: Dict[str, object] | None = None) -> SimulationResult:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    start_date = dt.datetime.strptime(str(cfg["start_date"]), "%Y-%m-%d").date()
    end_date = dt.datetime.strptime(str(cfg["end_date"]), "%Y-%m-%d").date()
    if cfg.get("vol_model") == "heston":
        prices = simulate_prices_heston(
            start_price=float(cfg["start_price"]),
            start_date=start_date,
            end_date=end_date,
            mu=float(cfg["mu"]),
            kappa=float(cfg["kappa"]),
            theta=float(cfg["theta"]),
            xi=float(cfg["xi"]),
            rho=float(cfg["rho"]),
            v0=float(cfg["v0"]),
            seed=int(cfg.get("seed", 0)),
        )
    else:
        prices = simulate_prices(
            start_price=float(cfg["start_price"]),
            start_date=start_date,
            end_date=end_date,
            mu=float(cfg["mu"]),
            sigma=float(cfg["sigma"]),
            seed=int(cfg.get("seed", 0)),
        )
    prices["EMA"] = compute_ema(prices["Close"], int(cfg["ema_period"]))

    capital = 10000.0
    equity = []
    trades: List[Trade] = []
    open_trades: List[Trade] = []
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    spacing = float(cfg["option_spacing"])
    dist_low, dist_high = cfg["short_leg_distance"]
    width_low, width_high = cfg["spread_width_range"]
    credit_ratio = float(cfg["trade_credit_ratio"])
    expiry_days = int(cfg["expiry_days"])
    risk_frac = float(cfg["risk_fraction"])
    max_lots = int(cfg["max_lots"])

    for idx, row in prices.iterrows():
        current_date = row["Date"]
        price = float(row["Close"])
        ema = float(row["EMA"])

        # expire trades first
        remaining = []
        for t in open_trades:
            if current_date >= t.expiry_date:
                if t.trade_type == "bull_put":
                    breached = price <= t.short_strike
                else:
                    breached = price >= t.short_strike
                if breached:
                    capital -= t.width * 100 * t.lots
                    t.result = -((t.width * 100 - t.credit) * t.lots)
                else:
                    t.result = t.credit * t.lots
                trades.append(t)
            else:
                remaining.append(t)
        open_trades = remaining

        # log equity after expirations
        equity.append((current_date, capital))

        # trading decision on Mondays only
        if current_date.weekday() == 0 and idx >= cfg["ema_period"]:
            trade_type = "bear_call" if price > ema else "bull_put"
            dist = rng.uniform(dist_low, dist_high)
            width = rng.integers(width_low, width_high + 1)
            if trade_type == "bull_put":
                short_strike = price - dist
                short_strike = np.floor(short_strike / spacing) * spacing
                long_strike = short_strike - width
            else:
                short_strike = price + dist
                short_strike = np.ceil(short_strike / spacing) * spacing
                long_strike = short_strike + width
            credit = credit_ratio * width * 100
            max_loss_per_lot = width * 100 - credit
            budget = capital * risk_frac
            lots = int(min(max_lots, np.floor(budget / max_loss_per_lot)))
            if lots >= 1:
                capital += credit * lots
                t = Trade(
                    open_date=current_date,
                    expiry_date=current_date + dt.timedelta(days=expiry_days),
                    trade_type=trade_type,
                    short_strike=round(short_strike, 2),
                    long_strike=round(long_strike, 2),
                    width=float(width),
                    credit=credit,
                    lots=lots,
                )
                open_trades.append(t)
    # finalize any trades after last date
    last_price = float(prices.iloc[-1]["Close"])
    final_date = prices.iloc[-1]["Date"]
    for t in open_trades:
        if t.trade_type == "bull_put":
            breached = last_price <= t.short_strike
        else:
            breached = last_price >= t.short_strike
        if breached:
            capital -= t.width * 100 * t.lots
            t.result = -((t.width * 100 - t.credit) * t.lots)
        else:
            t.result = t.credit * t.lots
        trades.append(t)
        equity.append((final_date, capital))

    equity_df = pd.DataFrame(equity, columns=["Date", "Equity"]).drop_duplicates("Date").reset_index(drop=True)
    returns = equity_df["Equity"].pct_change().fillna(0)
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    dd = (cumulative - peak) / peak
    max_dd = dd.min()
    wins = sum(1 for t in trades if t.result and t.result > 0)
    win_rate = wins / len(trades) if trades else 0.0
    total_return = (equity_df["Equity"].iloc[-1] - 10000.0) / 10000.0

    return SimulationResult(
        equity_curve=equity_df,
        trades=trades,
        total_return=total_return,
        win_rate=win_rate,
        max_drawdown=abs(max_dd),
    )


__all__ = [
    "simulate_prices",
    "simulate_prices_heston",
    "heston_path",
    "compute_ema",
    "run_simulation",
    "Trade",
    "SimulationResult",
]


def main(argv: list[str] | None = None) -> None:
    import argparse
    import os
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Synthetic credit-spread simulator")
    parser.add_argument("--start_price", type=float)
    parser.add_argument("--start_date")
    parser.add_argument("--end_date")
    parser.add_argument("--mu", type=float)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--ema_period", type=int)
    parser.add_argument("--risk_fraction", type=float)
    parser.add_argument("--max_lots", type=int)
    parser.add_argument("--short_leg_distance", nargs=2, type=float)
    parser.add_argument("--spread_width_range", nargs=2, type=int)
    parser.add_argument("--trade_credit_ratio", type=float)
    parser.add_argument("--option_spacing", type=float)
    parser.add_argument("--expiry_days", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--save_csv")

    args = parser.parse_args(argv)
    overrides: dict[str, object] = {}
    for key in [
        "start_price",
        "start_date",
        "end_date",
        "mu",
        "sigma",
        "ema_period",
        "risk_fraction",
        "max_lots",
        "short_leg_distance",
        "spread_width_range",
        "trade_credit_ratio",
        "option_spacing",
        "expiry_days",
        "seed",
    ]:
        val = getattr(args, key)
        if val is not None:
            overrides[key] = val

    if "short_leg_distance" in overrides:
        overrides["short_leg_distance"] = tuple(float(x) for x in overrides["short_leg_distance"])
    if "spread_width_range" in overrides:
        overrides["spread_width_range"] = tuple(int(x) for x in overrides["spread_width_range"])

    result = run_simulation(overrides)

    end_eq = result.equity_curve["Equity"].iloc[-1]
    print("--- Simulation Summary ---")
    print(f"End Equity:     ${end_eq:,.2f}")
    print(f"Total Return:   {result.total_return * 100:+.2f}%")
    print(f"Win Rate:       {result.win_rate * 100:.1f}%")
    print(f"Max Drawdown:   {result.max_drawdown * 100:.1f}%")
    print(f"Trades Placed:  {len(result.trades)}")

    if args.save_csv:
        os.makedirs(args.save_csv, exist_ok=True)
        result.equity_curve.to_csv(os.path.join(args.save_csv, "equity.csv"), index=False)
        pd.DataFrame([t.__dict__ for t in result.trades]).to_csv(
            os.path.join(args.save_csv, "trades.csv"), index=False
        )

    if not args.no_plot:
        plt.plot(result.equity_curve["Date"], result.equity_curve["Equity"])
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.title("Equity Curve")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

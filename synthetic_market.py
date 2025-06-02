from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
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
    start: dt.date,
    end: dt.date,
    mu: float,
    sigma: float,
    seed: int = 0,
) -> pd.DataFrame:
    """Return DataFrame of simulated prices using geometric Brownian motion."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    dt_frac = 1 / 252.0
    prices = np.empty(n)
    prices[0] = start_price
    for i in range(1, n):
        z = rng.standard_normal()
        prices[i] = prices[i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt_frac + sigma * np.sqrt(dt_frac) * z)
    return pd.DataFrame({"Date": dates.date, "Close": prices})


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Return exponential moving average for ``series``."""
    return series.ewm(span=period, adjust=False).mean()


def run_simulation(config: Dict[str, object] | None = None) -> SimulationResult:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    start = dt.datetime.strptime(str(cfg["start_date"]), "%Y-%m-%d").date()
    end = dt.datetime.strptime(str(cfg["end_date"]), "%Y-%m-%d").date()
    prices = simulate_prices(
        start_price=float(cfg["start_price"]),
        start=start,
        end=end,
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


__all__ = ["simulate_prices", "compute_ema", "run_simulation", "Trade", "SimulationResult"]

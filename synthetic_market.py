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
    commission: float = 0.0
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
    "credit_curve": False,
    "commissions": False,
    "option_spacing": 1,
    "expiry_days": 10,
    "seed": 123,
    # optional price dynamics parameters
    "ar1": 0.0,  # autocorrelation coefficient
    "vol_cluster": 0.0,  # volatility clustering weight
    "reaction": 1.0,  # nonlinear reaction exponent
}


def simulate_prices(
    start_price: float,
    start_date: dt.date,
    end_date: dt.date,
    mu: float,
    sigma: float,
    seed: int = 0,
    ar1: float = 0.0,
    vol_cluster: float = 0.0,
    reaction: float = 1.0,
) -> pd.DataFrame:
    """Return DataFrame of simulated prices.

    The base process is geometric Brownian motion. Optional parameters allow
    adding AR(1) autocorrelation (``ar1``), volatility clustering via a simple
    GARCH(1,1)-like term (``vol_cluster``), and a nonlinear price reaction
    exponent (``reaction``).
    """

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")
    n = len(dates)
    dt_frac = 1 / 252.0
    prices = np.empty(n)
    prices[0] = start_price
    prev_ret = 0.0
    variance = sigma ** 2
    for i in range(1, n):
        # update volatility based on previous return if clustering enabled
        if vol_cluster > 0:
            variance = (1 - vol_cluster) * (sigma ** 2) + vol_cluster * (prev_ret ** 2)
            sigma_t = np.sqrt(variance)
        else:
            sigma_t = sigma

        z = rng.standard_normal()
        ret = (mu - 0.5 * sigma_t ** 2) * dt_frac + sigma_t * np.sqrt(dt_frac) * z

        # add AR(1) autocorrelation
        if ar1 != 0.0:
            ret += ar1 * prev_ret

        # apply nonlinear reaction
        if reaction != 1.0:
            sign = np.sign(ret)
            ret = sign * (abs(ret) ** reaction)

        prices[i] = prices[i - 1] * np.exp(ret)
        prev_ret = ret

    return pd.DataFrame({"Date": dates.date, "Close": prices})


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Return exponential moving average for ``series``."""
    return series.ewm(span=period, adjust=False).mean()


def credit_ratio(dist: float, width: float) -> float:
    """Return credit ratio based on distance and width.

    Linearly fades from 0.45 at the money to 0.15 for very far strikes.
    """
    ratio = 0.45 - 0.3 * min(dist / max(width, 1.0), 1.0)
    return float(np.clip(ratio, 0.15, 0.45))


def run_simulation(config: Dict[str, object] | None = None) -> SimulationResult:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    start_date = dt.datetime.strptime(str(cfg["start_date"]), "%Y-%m-%d").date()
    end_date = dt.datetime.strptime(str(cfg["end_date"]), "%Y-%m-%d").date()
    prices = simulate_prices(
        start_price=float(cfg["start_price"]),
        start_date=start_date,
        end_date=end_date,
        mu=float(cfg["mu"]),
        sigma=float(cfg["sigma"]),
        seed=int(cfg.get("seed", 0)),
        ar1=float(cfg.get("ar1", 0.0)),
        vol_cluster=float(cfg.get("vol_cluster", 0.0)),
        reaction=float(cfg.get("reaction", 1.0)),
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
    credit_ratio_fixed = float(cfg["trade_credit_ratio"])
    use_curve = bool(cfg.get("credit_curve", False))
    use_commissions = bool(cfg.get("commissions", False))
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
                close_comm = t.commission if use_commissions else 0.0
                if breached:
                    capital -= t.width * 100 * t.lots + close_comm
                    t.result = -((t.width * 100 - t.credit) * t.lots) - (
                        2 * t.commission if use_commissions else 0.0
                    )
                else:
                    capital -= close_comm
                    t.result = t.credit * t.lots - (
                        2 * t.commission if use_commissions else 0.0
                    )
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
            ratio_val = credit_ratio(dist, width) if use_curve else credit_ratio_fixed
            credit = ratio_val * width * 100
            max_loss_per_lot = width * 100 - credit
            budget = capital * risk_frac
            lots = int(min(max_lots, np.floor(budget / max_loss_per_lot)))
            if lots >= 1:
                commission = 1.5 + 0.65 * (lots * 2) if use_commissions else 0.0
                capital += credit * lots - commission
                t = Trade(
                    open_date=current_date,
                    expiry_date=current_date + dt.timedelta(days=expiry_days),
                    trade_type=trade_type,
                    short_strike=round(short_strike, 2),
                    long_strike=round(long_strike, 2),
                    width=float(width),
                    credit=credit,
                    lots=lots,
                    commission=commission,
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
        close_comm = t.commission if use_commissions else 0.0
        if breached:
            capital -= t.width * 100 * t.lots + close_comm
            t.result = -((t.width * 100 - t.credit) * t.lots) - (
                2 * t.commission if use_commissions else 0.0
            )
        else:
            capital -= close_comm
            t.result = t.credit * t.lots - (
                2 * t.commission if use_commissions else 0.0
            )
        trades.append(t)
        equity.append((final_date, capital))

    equity_df = (
        pd.DataFrame(equity, columns=["Date", "Equity"]).drop_duplicates("Date", keep="last").reset_index(drop=True)
    )
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


def run_simulation_multi(
    config: Dict[str, object] | None = None,
    seeds: List[int] | None = None,
) -> pd.DataFrame:
    """Run ``run_simulation`` across multiple seeds and return a DataFrame of summary stats."""
    seeds = seeds or list(range(20))
    rows = []
    for s in seeds:
        cfg = {**(config or {}), "seed": s}
        res = run_simulation(cfg)
        rows.append({
            "seed": s,
            "total_return": res.total_return,
            "max_drawdown": res.max_drawdown,
            "win_rate": res.win_rate,
        })
    return pd.DataFrame(rows)


__all__ = [
    "simulate_prices",
    "compute_ema",
    "credit_ratio",
    "run_simulation",
    "run_simulation_multi",
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
    parser.add_argument("--credit_curve", action="store_true")
    parser.add_argument("--commissions", action="store_true")
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
        "credit_curve",
        "commissions",
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

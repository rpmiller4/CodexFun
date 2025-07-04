from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
import random

from option_analysis import black_scholes_price


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
    margin: float = 0.0
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
    "stop_mult": 2.0,
    "ga_ensemble_n": 5,
    "ga_seed_mode": "mean",
    "seed": 123,
    # optional price dynamics parameters
    "ar1": 0.0,  # autocorrelation coefficient
    "vol_cluster": 0.0,  # volatility clustering weight
    "reaction": 1.0,  # nonlinear reaction exponent
    "liq_decay": 12,
    "use_heston": False,
    "heston": {
        "v0": 0.04,
        "kappa": 1.5,
        "theta": 0.04,
        "xi": 0.5,
        "rho": -0.7,
    },
    "realism": {
        "slippage": False,
        "liquidity": False,
        "margin": False,
        "jumps": False,
        "iv_pricing": False,
        "mtm": False,
        "crash_test": False,
    },
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
    jump_prob: float = 0.0,
    down_jump: float = -0.08,
    up_jump: float = 0.04,
) -> pd.DataFrame:
    """Return DataFrame of simulated prices.

    The base process is geometric Brownian motion. Optional parameters allow
    adding AR(1) autocorrelation (``ar1``), volatility clustering via a simple
    GARCH(1,1)-like term (``vol_cluster``), a nonlinear price reaction
    exponent (``reaction``) and rare jump-diffusion events controlled by
    ``jump_prob`` and jump magnitudes.
    """

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")
    n = len(dates)
    dt_frac = 1 / 252.0
    prices = np.empty(n)
    jumps = np.zeros(n, dtype=bool)
    prices[0] = start_price
    prev_ret = 0.0
    variance = sigma**2
    for i in range(1, n):
        # update volatility based on previous return if clustering enabled
        if vol_cluster > 0:
            variance = (1 - vol_cluster) * (sigma**2) + vol_cluster * (prev_ret**2)
            sigma_t = np.sqrt(variance)
        else:
            sigma_t = sigma

        z = rng.standard_normal()
        ret = (mu - 0.5 * sigma_t**2) * dt_frac + sigma_t * np.sqrt(dt_frac) * z

        # add AR(1) autocorrelation
        if ar1 != 0.0:
            ret += ar1 * prev_ret

        # apply nonlinear reaction
        if reaction != 1.0:
            sign = np.sign(ret)
            ret = sign * (abs(ret) ** reaction)

        price = prices[i - 1] * np.exp(ret)

        jump_flag = False
        if jump_prob > 0 and rng.uniform() < jump_prob:
            jump_flag = True
            move = rng.choice([down_jump, up_jump], p=[2 / 3, 1 / 3])
            price *= 1 + move

        prices[i] = price
        jumps[i] = jump_flag
        prev_ret = ret

    return pd.DataFrame({"Date": dates.date, "Close": prices, "Jump": jumps})


def simulate_heston_prices(cfg: Dict[str, object]) -> pd.DataFrame:
    """Return DataFrame of prices/variance following the Heston model."""
    start_date = dt.datetime.strptime(str(cfg["start_date"]), "%Y-%m-%d").date()
    end_date = dt.datetime.strptime(str(cfg["end_date"]), "%Y-%m-%d").date()
    dates = pd.date_range(start_date, end_date, freq="D")
    n = len(dates)
    dt_frac = 1 / 252.0
    h = cfg.get("heston", {})
    s0 = float(cfg.get("start_price", 100.0))
    v0 = float(h.get("v0", 0.04))
    kappa = float(h.get("kappa", 1.5))
    theta = float(h.get("theta", 0.04))
    xi = float(h.get("xi", 0.5))
    rho = float(h.get("rho", -0.7))
    mu = float(cfg.get("mu", 0.0))
    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)

    prices = np.empty(n)
    variances = np.empty(n)
    prices[0] = s0
    variances[0] = v0
    for i in range(1, n):
        z1 = rng.standard_normal()
        z2 = rng.standard_normal()
        z2 = rho * z1 + math.sqrt(1 - rho**2) * z2
        v_prev = max(variances[i - 1], 0.0)
        v = (
            v_prev
            + kappa * (theta - v_prev) * dt_frac
            + xi * math.sqrt(v_prev) * math.sqrt(dt_frac) * z2
        )
        v = max(v, 0.0)
        variances[i] = v
        ret = (mu - 0.5 * v_prev) * dt_frac + math.sqrt(v_prev) * math.sqrt(
            dt_frac
        ) * z1
        prices[i] = prices[i - 1] * math.exp(ret)

    return pd.DataFrame({"Date": dates.date, "Close": prices, "Var": variances})


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Return exponential moving average for ``series``."""
    return series.ewm(span=period, adjust=False).mean()


def credit_ratio(dist: float, width: float) -> float:
    """Return credit ratio based on distance and width.

    Linearly fades from 0.45 at the money to 0.15 for very far strikes.
    """
    ratio = 0.45 - 0.3 * min(dist / max(width, 1.0), 1.0)
    return float(np.clip(ratio, 0.15, 0.45))


def quote_spread(distance_pts: float) -> float:
    """Return bid/ask spread in dollars based on strike distance."""
    return 0.02 + 0.01 * (distance_pts / 5) ** 2


def imp_vol(distance_pts: float, base_sigma: float) -> float:
    """Return crude implied volatility with simple smile.

    Positive ``distance_pts`` denotes put strikes (rich), negative denotes
    call strikes (cheap).
    """
    sign = np.sign(distance_pts) or 1.0
    return base_sigma * (1.0 + 0.25 * sign)


def _bs_cached(
    cache: Dict[tuple, float],
    S: float,
    K: float,
    T: float,
    sigma: float,
    option_type: str,
) -> float:
    """Return Black-Scholes price with simple memoization."""
    key = (round(S, 2), round(K, 2), round(T, 4), round(sigma, 4), option_type)
    if key not in cache:
        cache[key] = black_scholes_price(S, K, T, 0.05, sigma, option_type)
    return cache[key]


def run_simulation(config: Dict[str, object] | None = None) -> SimulationResult:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    realism_cfg = {**DEFAULT_CONFIG["realism"], **cfg.get("realism", {})}
    cfg["realism"] = realism_cfg
    start_date = dt.datetime.strptime(str(cfg["start_date"]), "%Y-%m-%d").date()
    end_date = dt.datetime.strptime(str(cfg["end_date"]), "%Y-%m-%d").date()
    if cfg.get("use_heston", False):
        prices = simulate_heston_prices(cfg)
    else:
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
            jump_prob=(
                float(realism_cfg.get("jump_prob", 0.01))
                if realism_cfg.get("jumps")
                else 0.0
            ),
            down_jump=float(realism_cfg.get("down_jump", -0.08)),
            up_jump=float(realism_cfg.get("up_jump", 0.04)),
        )
    prices["EMA"] = compute_ema(prices["Close"], int(cfg["ema_period"]))

    capital = 10000.0
    equity = []
    trades: List[Trade] = []
    open_trades: List[Trade] = []
    blocked_margin = 0.0
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
    base_sigma = float(cfg["sigma"])
    bs_cache: Dict[tuple, float] = {}
    use_var = "Var" in prices.columns

    for idx, row in prices.iterrows():
        current_date = row["Date"]
        price = float(row["Close"])
        ema = float(row["EMA"])
        sigma_row = math.sqrt(row["Var"]) if use_var else base_sigma

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
                if realism_cfg.get("margin"):
                    blocked_margin -= t.margin
                trades.append(t)
            else:
                remaining.append(t)
        open_trades = remaining

        # mark-to-market open positions
        if realism_cfg.get("mtm"):
            mtm_list = []
            for t in open_trades:
                T = max((t.expiry_date - current_date).days, 0) / 365.0
                opt_type = "put" if t.trade_type == "bull_put" else "call"
                iv_s = imp_vol(price - t.short_strike, sigma_row)
                iv_l = imp_vol(price - t.long_strike, sigma_row)
                s_p = _bs_cached(bs_cache, price, t.short_strike, T, iv_s, opt_type)
                l_p = _bs_cached(bs_cache, price, t.long_strike, T, iv_l, opt_type)
                spread_val = (s_p - l_p) * 100 * t.lots
                pnl = t.credit * t.lots - spread_val
                loss = -pnl if pnl < 0 else 0.0
                if loss >= float(cfg.get("stop_mult", 2.0)) * t.credit * t.lots:
                    close_comm = t.commission if use_commissions else 0.0
                    capital += pnl - close_comm
                    if realism_cfg.get("margin"):
                        blocked_margin -= t.margin
                    t.result = pnl - (2 * t.commission if use_commissions else 0.0)
                    trades.append(t)
                    continue
                if (t.trade_type == "bull_put" and price <= t.short_strike) or (
                    t.trade_type == "bear_call" and price >= t.short_strike
                ):
                    intrinsic = (
                        t.short_strike - price
                        if t.trade_type == "bull_put"
                        else price - t.short_strike
                    )
                    intrinsic = max(intrinsic, 0) * 100 * t.lots
                    time_val = max(spread_val - intrinsic, 0.0)
                    prob = min(0.1, time_val / spread_val) if spread_val > 0 else 0.0
                    if rng.uniform() < prob:
                        close_comm = t.commission if use_commissions else 0.0
                        capital -= (
                            t.width * 100 * t.lots - t.credit * t.lots
                        ) + close_comm
                        if realism_cfg.get("margin"):
                            blocked_margin -= t.margin
                        t.result = -((t.width * 100 - t.credit) * t.lots) - (
                            2 * t.commission if use_commissions else 0.0
                        )
                        trades.append(t)
                        continue
                mtm_list.append(t)
            open_trades = mtm_list
            unrealized = 0.0
            for t in open_trades:
                T = max((t.expiry_date - current_date).days, 0) / 365.0
                opt_type = "put" if t.trade_type == "bull_put" else "call"
                iv_s = imp_vol(price - t.short_strike, sigma_row)
                iv_l = imp_vol(price - t.long_strike, sigma_row)
                s_p = _bs_cached(bs_cache, price, t.short_strike, T, iv_s, opt_type)
                l_p = _bs_cached(bs_cache, price, t.long_strike, T, iv_l, opt_type)
                spread_val = (s_p - l_p) * 100 * t.lots
                unrealized += t.credit * t.lots - spread_val
            equity_val = capital + unrealized
        else:
            equity_val = capital

        equity.append((current_date, equity_val))

        # trading decision on Mondays only
        if current_date.weekday() == 0 and idx >= cfg["ema_period"]:
            trade_type = "bear_call" if price > ema else "bull_put"
            dist = rng.uniform(dist_low, dist_high)
            width = rng.integers(width_low, width_high + 1)
            if realism_cfg.get("liquidity"):
                p_quote = float(np.exp(-dist / float(cfg.get("liq_decay", 12))))
                if rng.uniform() > p_quote:
                    continue
            if trade_type == "bull_put":
                short_strike = price - dist
                short_strike = np.floor(short_strike / spacing) * spacing
                long_strike = short_strike - width
            else:
                short_strike = price + dist
                short_strike = np.ceil(short_strike / spacing) * spacing
                long_strike = short_strike + width
            if realism_cfg.get("iv_pricing"):
                T = expiry_days / 365.0
                opt_type = "put" if trade_type == "bull_put" else "call"
                iv_s = imp_vol(price - short_strike, sigma_row)
                iv_l = imp_vol(price - long_strike, sigma_row)
                short_p = _bs_cached(bs_cache, price, short_strike, T, iv_s, opt_type)
                long_p = _bs_cached(bs_cache, price, long_strike, T, iv_l, opt_type)
                credit = (short_p - long_p) * 100
                if realism_cfg.get("slippage"):
                    credit -= 0.5 * quote_spread(dist) * rng.uniform() * 100
            else:
                ratio_val = (
                    credit_ratio(dist, width) if use_curve else credit_ratio_fixed
                )
                credit = ratio_val * width * 100
                if realism_cfg.get("slippage"):
                    mid = credit / 100
                    fill = mid - 0.5 * quote_spread(dist) * rng.uniform()
                    credit = fill * 100
            max_loss_per_lot = width * 100 - credit
            budget = capital * risk_frac
            lots = int(min(max_lots, np.floor(budget / max_loss_per_lot)))
            if lots >= 1:
                margin_req = max(width * 100 - credit, 0) * lots
                if (
                    realism_cfg.get("margin")
                    and capital - blocked_margin - margin_req < 0
                ):
                    continue
                commission = 1.5 + 0.65 * (lots * 2) if use_commissions else 0.0
                capital += credit * lots - commission
                if realism_cfg.get("margin"):
                    blocked_margin += margin_req
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
                    margin=margin_req if realism_cfg.get("margin") else 0.0,
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
        if realism_cfg.get("margin"):
            blocked_margin -= t.margin
        trades.append(t)
        equity.append((final_date, capital))

    equity_df = (
        pd.DataFrame(equity, columns=["Date", "Equity"])
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
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
        rows.append(
            {
                "seed": s,
                "total_return": res.total_return,
                "max_drawdown": res.max_drawdown,
                "win_rate": res.win_rate,
            }
        )
    return pd.DataFrame(rows)


def _seed_list(cfg: Dict[str, object], n: int) -> List[int]:
    """Return deterministic seed list derived from ``cfg``."""
    import hashlib

    items = [(k, cfg[k]) for k in sorted(cfg) if k != "seed"]
    rep = repr(items).encode()
    h = int(hashlib.sha256(rep).hexdigest(), 16)
    rng = random.Random(h)
    return [rng.randrange(0, 2**32 - 1) for _ in range(n)]


def ga_fitness(config: Dict[str, object]) -> float:
    """Return GA fitness using an ensemble of seeds."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    n = int(cfg.get("ga_ensemble_n", DEFAULT_CONFIG.get("ga_ensemble_n", 5)))
    mode = str(cfg.get("ga_seed_mode", DEFAULT_CONFIG.get("ga_seed_mode", "mean")))
    cfg = {k: v for k, v in cfg.items() if k != "seed"}
    seeds = _seed_list(cfg, n)
    df = run_simulation_multi(cfg, seeds=seeds)
    scores = df["total_return"] - df["max_drawdown"]
    return float(scores.mean() if mode == "mean" else scores.min())


def generate_crash_path(start_price: float, days: int) -> pd.DataFrame:
    """Return deterministic crash path: -35% over 30 days with 80% vol."""
    dates = pd.date_range(dt.date.today(), periods=days, freq="D")
    prices = []
    for i in range(days):
        if i < 30:
            frac = (i + 1) / 30.0
            price = start_price * (1 - 0.35 * frac)
        else:
            last = prices[-1] if prices else start_price
            sigma = 0.80
            ret = np.random.normal(-0.5 * sigma**2 / 252, sigma / math.sqrt(252))
            price = last * math.exp(ret)
        prices.append(price)
    return pd.DataFrame({"Date": dates.date, "Close": prices})


__all__ = [
    "simulate_prices",
    "simulate_heston_prices",
    "compute_ema",
    "credit_ratio",
    "quote_spread",
    "imp_vol",
    "ga_fitness",
    "generate_crash_path",
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
        overrides["short_leg_distance"] = tuple(
            float(x) for x in overrides["short_leg_distance"]
        )
    if "spread_width_range" in overrides:
        overrides["spread_width_range"] = tuple(
            int(x) for x in overrides["spread_width_range"]
        )

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
        result.equity_curve.to_csv(
            os.path.join(args.save_csv, "equity.csv"), index=False
        )
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

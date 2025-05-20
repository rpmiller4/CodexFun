# -*- coding: utf-8 -*-
"""Utilities for analysing SPY option contracts."""

from __future__ import annotations

import datetime as dt
import math
from typing import List, Optional

import numpy as np

from models import OptionContract, OptionAnalysis

import yfinance as yf


def _norm_cdf(x: float) -> float:
    """Cumulative distribution function for the standard normal distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _valid_iv(iv: float | None, floor: float = 0.01) -> bool:
    """Return True if iv is usable (not None/NaN/zero and ≥ floor)."""
    return (iv is not None) and (iv >= floor) and (not math.isnan(iv))


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Return Black-Scholes option price."""
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def black_scholes_d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Return the d2 term used in Black-Scholes."""
    if T <= 0:
        return -float("inf")
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return d1 - sigma * math.sqrt(T)


def get_expiries_by_market_days(targets: List[int] = [3, 7, 14, 21]) -> List[str]:
    """Return expiry strings matching the provided market-day targets."""
    ticker = yf.Ticker("SPY")
    expiries = ticker.options
    today = dt.date.today()
    expiry_dates = [dt.datetime.strptime(e, "%Y-%m-%d").date() for e in expiries]

    selected: List[str] = []
    for target in targets:
        chosen = None
        for exp in expiry_dates:
            if np.busday_count(today, exp) >= target:
                chosen = exp
                break
        if chosen is not None:
            selected.append(chosen.strftime("%Y-%m-%d"))
    return selected




def compute_option_metrics(
    option: OptionContract, r: float = 0.05, today: Optional[dt.date] = None
) -> OptionAnalysis:
    """Return computed metrics for a given option contract."""
    if today is None:
        today = dt.date.today()
    days_to_expiry = max((option.expiry - today).days, 0)
    T = days_to_expiry / 365.0

    d2 = black_scholes_d2(option.underlying_price, option.strike, T, r, option.iv)
    pop = _norm_cdf(d2)

    if option.option_type == "call":
        intrinsic = max(option.underlying_price - option.strike, 0.0)
    else:
        intrinsic = max(option.strike - option.underlying_price, 0.0)
    time_value = max(option.last_price - intrinsic, 0.0)

    return OptionAnalysis(
        strike=option.strike,
        last_price=option.last_price,
        iv=option.iv,
        expiry=option.expiry,
        underlying_price=option.underlying_price,
        pop=pop,
        intrinsic_value=intrinsic,
        time_value=time_value,
        days_to_expiry=days_to_expiry,
    )


def get_call_option_analysis(expiry_strs: List[str], r: float = 0.05) -> List[OptionAnalysis]:
    """Fetch SPY call options for the given expiries and compute metrics."""
    ticker = yf.Ticker("SPY")
    underlying = float(ticker.history(period="1d")["Close"].iloc[-1])

    results: List[OptionAnalysis] = []
    today = dt.date.today()
    for expiry_str in expiry_strs:
        expiry_date = dt.datetime.strptime(expiry_str, "%Y-%m-%d").date()
        chain = ticker.option_chain(expiry_str).calls
        days_to_expiry = max((expiry_date - today).days, 0)
        T = days_to_expiry / 365.0
        for _, row in chain.iterrows():
            strike = float(row["strike"])
            price = float(row["lastPrice"])
            iv = float(row.get("impliedVolatility", 0.0))
            if not _valid_iv(iv):
                continue
            d2 = black_scholes_d2(underlying, strike, T, r, iv)
            pop = _norm_cdf(d2)
            intrinsic = max(underlying - strike, 0.0)
            time_value = max(price - intrinsic, 0.0)
            results.append(
                OptionAnalysis(
                    strike=strike,
                    last_price=price,
                    iv=iv,
                    expiry=expiry_date,
                    underlying_price=underlying,
                    pop=pop,
                    intrinsic_value=intrinsic,
                    time_value=time_value,
                    days_to_expiry=days_to_expiry,
                )
            )
    return results


def filter_options_by_pop_and_timevalue_percent(
    options: List[OptionAnalysis],
    pop_threshold: float = 0.65,
    tv_threshold: float = 0.25,
) -> List[OptionAnalysis]:
    """Filter options by probability of profit and time value percentage."""
    filtered = []
    for opt in options:
        if opt.last_price == 0:
            continue
        tv_percent = opt.time_value / opt.last_price
        if opt.pop > pop_threshold and tv_percent > tv_threshold:
            filtered.append(opt)
    filtered.sort(key=lambda o: o.pop, reverse=True)
    return filtered


def simulate_value_decay(
    option: OptionContract, days: List[int], r: float = 0.05
) -> List[float]:
    """Return theoretical option prices as time to expiry decreases."""
    prices: List[float] = []
    for d in days:
        T = max(d / 365.0, 0.0)
        price = black_scholes_price(
            option.underlying_price,
            option.strike,
            T,
            r,
            option.iv,
            option.option_type,
        )
        prices.append(price)
    return prices

__all__ = [
    "OptionContract",
    "OptionAnalysis",
    "black_scholes_price",
    "compute_option_metrics",
    "get_expiries_by_market_days",
    "get_call_option_analysis",
    "filter_options_by_pop_and_timevalue_percent",
    "simulate_value_decay",
]


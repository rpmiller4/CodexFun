# -*- coding: utf-8 -*-
"""Utilities for analysing SPY option contracts."""

from __future__ import annotations

import datetime as dt
import math
from typing import List, Optional

from models import OptionContract, OptionAnalysis

import yfinance as yf


def _norm_cdf(x: float) -> float:
    """Cumulative distribution function for the standard normal distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


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




def compute_option_metrics(
    option: OptionContract, r: float = 0.05, today: Optional[dt.date] = None
) -> OptionAnalysis:
    """Return computed metrics for a given option contract."""
    if today is None:
        today = dt.date.today()
    T = max((option.expiry - today).days / 365.0, 0.0)

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
    )


def get_call_option_analysis(r: float = 0.05) -> List[OptionAnalysis]:
    """Fetch SPY call options for the nearest expiry and compute metrics."""
    ticker = yf.Ticker("SPY")
    expiries = ticker.options
    if not expiries:
        raise RuntimeError("No expiries found for SPY")
    expiry_str = expiries[0]
    expiry_date = dt.datetime.strptime(expiry_str, "%Y-%m-%d").date()

    underlying = float(ticker.history(period="1d")["Close"].iloc[-1])
    chain = ticker.option_chain(expiry_str).calls

    results: List[OptionAnalysis] = []
    T = max((expiry_date - dt.date.today()).days / 365.0, 0.0)
    for _, row in chain.iterrows():
        strike = float(row["strike"])
        price = float(row["lastPrice"])
        iv = float(row.get("impliedVolatility", 0.0))
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
            )
        )
    return results


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
    "get_call_option_analysis",
    "simulate_value_decay",
]


from __future__ import annotations

import math
from enum import Enum

import numpy as np
import yfinance as yf

from utils import capped_sigma, fetch_with_retry, MIN_SIGMA

MIN_IV_DEFAULT = 0.05  # 5 %

class IVSource(str, Enum):
    ORIG = "orig"
    ATM = "atm"
    VIX = "vix"


def iv_is_valid(iv: float | None, floor: float = MIN_IV_DEFAULT) -> bool:
    """Return True if IV is usable (not None/NaN and >= floor)."""
    return iv is not None and iv >= floor and not math.isnan(iv)


def get_atm_iv(chain_df, spot):
    """Return IV of strike closest to spot; NaN if chain empty or IV missing."""
    if chain_df is None or len(chain_df) == 0:
        return float("nan")
    idx = (chain_df["strike"] - spot).abs().idxmin()
    return float(chain_df.loc[idx, "impliedVolatility"])


def vix_sigma(days: int, floor: float = MIN_SIGMA) -> float:
    """Convert ^VIX (%) to annual sigma scaled to days-to-expiry with floor."""
    vix_df = fetch_with_retry(yf.Ticker("^VIX").history, period="1d")
    vix_pct = vix_df["Close"].iloc[-1] / 100.0
    raw = vix_pct * math.sqrt(days / 30.0)
    return capped_sigma(raw, floor)


def resolve_iv(
    raw_iv: float,
    chain_df,
    spot: float,
    days: int,
    floor: float = MIN_IV_DEFAULT,
    sigma_floor: float = MIN_SIGMA,
):
    """Return (iv, IVSource) after fallback: raw -> ATM -> VIX."""
    if iv_is_valid(raw_iv, floor):
        return raw_iv, IVSource.ORIG
    iv = get_atm_iv(chain_df, spot)
    if iv_is_valid(iv, floor):
        return iv, IVSource.ATM
    iv = vix_sigma(days, floor=sigma_floor)
    return iv, IVSource.VIX

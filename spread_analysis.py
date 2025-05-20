from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import yfinance as yf

from models import OptionContract
import option_analysis as oa
from vol_utils import resolve_iv, iv_is_valid, MIN_IV_DEFAULT, IVSource


def _norm_cdf(x: float) -> float:
    """Cumulative normal distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class SpreadType(str, Enum):
    BULL_PUT = "bull_put"
    BEAR_CALL = "bear_call"


@dataclass
class Spread:
    type: SpreadType
    short: OptionContract
    long: OptionContract
    width: float
    credit: float
    max_loss: float
    credit_pct: float
    pop: float
    iv_short_src: IVSource
    iv_long_src: IVSource
    days_to_expiry: int


def make_spread(
    spread_type: SpreadType,
    short: OptionContract,
    long: OptionContract,
    r: float = 0.05,
    today: Optional[dt.date] = None,
) -> Spread:
    """Build a Spread object from two option legs."""
    if today is None:
        today = dt.date.today()
    width = abs(long.strike - short.strike)
    credit = short.last_price - long.last_price
    max_loss = width - credit
    # credit percentage relative to spread width
    credit_pct = credit / width * 100 if width else 0.0
    days_to_expiry = max((short.expiry - today).days, 0)
    T = days_to_expiry / 365.0
    d2 = oa.black_scholes_d2(short.underlying_price, short.strike, T, r, short.iv)
    if spread_type == SpreadType.BULL_PUT:
        pop = _norm_cdf(d2)
    else:
        pop = _norm_cdf(-d2)
    return Spread(
        type=spread_type,
        short=short,
        long=long,
        width=width,
        credit=credit,
        max_loss=max_loss,
        credit_pct=credit_pct,
        pop=pop,
        iv_short_src=short.iv_src,
        iv_long_src=long.iv_src,
        days_to_expiry=days_to_expiry,
    )


def get_credit_spreads(
    expiry: str,
    width: float = 5.0,
    spread_type: SpreadType = SpreadType.BULL_PUT,
    min_iv: float = MIN_IV_DEFAULT,
) -> List[Spread]:
    """Return credit spreads for the given expiry and width."""
    ticker = yf.Ticker("SPY")
    underlying = float(ticker.history(period="1d")["Close"].iloc[-1])
    chain = ticker.option_chain(expiry)
    df = chain.puts if spread_type == SpreadType.BULL_PUT else chain.calls
    expiry_date = dt.datetime.strptime(expiry, "%Y-%m-%d").date()
    today = dt.date.today()

    options = {}
    for _, row in df.iterrows():
        strike = float(row["strike"])
        iv = float(row.get("impliedVolatility", 0.0))
        opt = OptionContract(
            strike=strike,
            expiry=expiry_date,
            option_type="put" if spread_type == SpreadType.BULL_PUT else "call",
            last_price=float(row["lastPrice"]),
            iv=iv,
            iv_src=IVSource.ORIG,
            underlying_price=underlying,
        )
        options[strike] = opt

    spreads: List[Spread] = []
    days = max((expiry_date - today).days, 0)
    for strike, short in options.items():
        long_strike = strike - width if spread_type == SpreadType.BULL_PUT else strike + width
        long = options.get(long_strike)
        if long is None:
            continue
        ivS, srcS = resolve_iv(short.iv, df, underlying, days, min_iv)
        ivL, srcL = resolve_iv(long.iv, df, underlying, days, min_iv)
        if not (iv_is_valid(ivS, min_iv) and iv_is_valid(ivL, min_iv)):
            continue
        if short.last_price == 0 or long.last_price == 0:
            continue
        short_c = OptionContract(**{**short.__dict__, "iv": ivS, "iv_src": srcS})
        long_c = OptionContract(**{**long.__dict__, "iv": ivL, "iv_src": srcL})
        spreads.append(make_spread(spread_type, short_c, long_c, today=today))
    return spreads


def filter_credit_spreads(
    spreads: List[Spread],
    pop_min: float = 0.65,
    credit_min_pct: float = 25.0,
) -> List[Spread]:
    """Filter spreads by probability of profit and credit percentage."""
    filtered = [
        s
        for s in spreads
        if s.pop >= pop_min and s.credit_pct >= credit_min_pct
    ]
    filtered.sort(key=lambda s: (-s.pop, -s.credit_pct))
    return filtered


__all__ = [
    "SpreadType",
    "Spread",
    "make_spread",
    "get_credit_spreads",
    "filter_credit_spreads",
]

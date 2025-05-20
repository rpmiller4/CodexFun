from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

from vol_utils import IVSource

@dataclass
class OptionContract:
    """Generic option contract information."""

    strike: float
    expiry: dt.date
    option_type: str  # "call" or "put"
    last_price: float
    iv: float
    iv_src: IVSource
    underlying_price: float
    symbol: Optional[str] = None
    open_interest: Optional[int] = None


@dataclass
class OptionAnalysis:
    """Computed statistics for an option contract."""

    strike: float
    last_price: float
    iv: float
    expiry: dt.date
    underlying_price: float
    pop: float
    intrinsic_value: float
    time_value: float
    days_to_expiry: int

import datetime as dt
from typing import List


def expiries_within(ticker, max_days: int = 14) -> List[str]:
    """Return expiry strings within max_days calendar days from today."""
    today = dt.date.today()
    valid: List[str] = []
    for exp in getattr(ticker, "options", []):
        try:
            d = dt.datetime.strptime(exp, "%Y-%m-%d").date()
        except Exception:
            continue
        if 0 < (d - today).days <= max_days:
            valid.append(exp)
    return sorted(valid)

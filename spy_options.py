"""Fetch latest SPY value and find best options contract.

This script uses yfinance to retrieve the most recent SPY price and
options chain data to identify an option contract that is near the
current price and has the highest open interest.

Note: Internet access is required to fetch data from Yahoo Finance.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

from models import OptionContract

import pandas as pd
import yfinance as yf




def get_latest_spy_price() -> float:
    """Return the latest SPY price using yfinance."""
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="1d")
    if data.empty:
        raise RuntimeError("No data returned for SPY")
    return float(data["Close"].iloc[-1])


def get_options_chain(expiry: str) -> pd.DataFrame:
    """Return the options chain for SPY for the given expiry date."""
    ticker = yf.Ticker("SPY")
    opt = ticker.option_chain(expiry)
    calls = opt.calls.assign(option_type="call")
    puts = opt.puts.assign(option_type="put")
    chain = pd.concat([calls, puts], ignore_index=True)
    return chain


def find_best_option(expiry: str, option_type: str = "call") -> Optional[OptionContract]:
    """Find the option near the money with highest open interest."""
    price = get_latest_spy_price()
    chain = get_options_chain(expiry)

    # Filter for desired option type
    chain = chain[chain["option_type"] == option_type]

    # Compute distance from current price
    chain["distance"] = (chain["strike"] - price).abs()

    # Sort by distance (ATM) then open interest descending
    chain = chain.sort_values(by=["distance", "openInterest"], ascending=[True, False])

    if chain.empty:
        return None

    best = chain.iloc[0]
    return OptionContract(
        symbol=best["contractSymbol"],
        strike=float(best["strike"]),
        expiry=dt.datetime.strptime(expiry, "%Y-%m-%d").date(),
        option_type=option_type,
        last_price=float(best["lastPrice"]),
        open_interest=int(best["openInterest"]),
    )


def list_option_expiries():
    """List available option expiry dates for SPY."""
    ticker = yf.Ticker("SPY")
    return ticker.options


def main() -> None:
    # Example usage
    expiries = list_option_expiries()
    if not expiries:
        raise RuntimeError("No expiries found for SPY")

    expiry = expiries[0]  # choose the nearest expiry
    best_call = find_best_option(expiry, option_type="call")
    if best_call:
        print("Latest SPY price:", get_latest_spy_price())
        print("Best call option:")
        print(best_call)
    else:
        print("No suitable option found")


if __name__ == "__main__":
    main()

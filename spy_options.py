"""Fetch latest SPY value and find best options contract.

This script uses yfinance to retrieve the most recent SPY price and
options chain data to identify an option contract that is near the
current price and has the highest open interest.

Note: Internet access is required to fetch data from Yahoo Finance.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional, List

import argparse
import csv

from models import OptionContract
import option_analysis as oa

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


def main(args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="SPY option analysis")
    parser.add_argument("--pop-threshold", type=float, default=0.65)
    parser.add_argument("--tv-threshold", type=float, default=0.25)
    parsed = parser.parse_args(args)

    expiries = oa.get_expiries_by_market_days([3, 7, 14, 21])
    all_options = oa.get_call_option_analysis(expiries)
    filtered = oa.filter_options_by_pop_and_timevalue_percent(
        all_options, parsed.pop_threshold, parsed.tv_threshold
    )

    groups = {}
    for opt in filtered:
        groups.setdefault(opt.days_to_expiry, []).append(opt)

    with open("filtered_options.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Strike", "PoP", "Time %", "IV", "Days"])
        for days in sorted(groups.keys()):
            for opt in groups[days]:
                tv_perc = opt.time_value / opt.last_price * 100 if opt.last_price else 0
                writer.writerow([
                    f"{opt.strike:.1f}",
                    f"{opt.pop:.2f}",
                    f"{tv_perc:.1f}%",
                    f"{opt.iv:.2f}",
                    days,
                ])

    for days in sorted(groups.keys()):
        print(f"### Expiry ~{days} Market Days")
        print("Strike | PoP | Time % | IV | Days")
        print("----------------------------------")
        for opt in groups[days]:
            tv_perc = opt.time_value / opt.last_price * 100 if opt.last_price else 0
            print(
                f"{opt.strike:.1f} | {opt.pop:.2f} | {tv_perc:.1f}% | {opt.iv:.2f} | {days}"
            )
        print()


if __name__ == "__main__":
    main()

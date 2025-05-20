from __future__ import annotations

import argparse
import datetime as dt
from typing import List, Optional

import yfinance as yf

import option_analysis as oa
from spread_analysis import SpreadType, get_credit_spreads, filter_credit_spreads
import expiry_selector


def main(args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="SPY credit-spread screener",
        epilog="Outputs headings like '=== Bull_Put Spreads ~7 Days | Width $5 ==='",
    )
    parser.add_argument(
        "--type",
        choices=[t.value for t in SpreadType],
        default=SpreadType.BULL_PUT.value,
    )
    parser.add_argument("--pop", type=float, default=0.65)
    parser.add_argument("--credit", dest="credit_pct", type=float, default=25.0)
    parser.add_argument("--widths", nargs="+", type=float, default=[5.0])
    parser.add_argument(
        "--expiry-dates",
        nargs="+",
        help="Explicit list of expiries (YYYY-MM-DD). Overrides --max-days",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=14,
        help="Analyze every expiry up to this many calendar days ahead (default 14)",
    )
    parser.add_argument(
        "--min-iv",
        type=float,
        default=0.05,
        help=(
            "Floor for acceptable implied volatility (default 0.05 = 5 %%).  "
            "Missing/low IV is replaced by ATM IV, else VIX \u03c3.  "
            "Spreads are skipped if resulting IV < floor."
        ),
    )
    parsed = parser.parse_args(args)

    spread_type = SpreadType(parsed.type)

    ticker = yf.Ticker("SPY")
    explicit_list = parsed.expiry_dates if parsed.expiry_dates else []
    expiries = (
        explicit_list
        if explicit_list
        else expiry_selector.expiries_within(ticker, max_days=parsed.max_days)
    )
    today = dt.date.today()

    for expiry in expiries:
        days = (
            dt.datetime.strptime(expiry, "%Y-%m-%d").date() - today
        ).days
        for width in parsed.widths:
            spreads = get_credit_spreads(
                expiry,
                width=width,
                spread_type=spread_type,
                min_iv=parsed.min_iv,
            )
            filtered = filter_credit_spreads(
                spreads, pop_min=parsed.pop, credit_min_pct=parsed.credit_pct
            )
            if not filtered:
                continue
            stype = spread_type.value
            print(
                f"=== {stype.title()} Spreads ~{days} Days | Width ${width} ==="
            )
            print(
                f"{'Short':>5} {'Long':>5} {'Credit':>7} {'MaxLoss':>8} {'Credit%':>8} {'PoP':>5} {'IVs':>7} {'Src':>9} {'Days':>4}"
            )
            for sp in filtered:
                ivs = f"{sp.short.iv:.2f}/{sp.long.iv:.2f}"
                srcs = f"{sp.iv_short_src.value}/{sp.iv_long_src.value}"
                print(
                    f"{sp.short.strike:5.0f} {sp.long.strike:5.0f} {sp.credit:7.2f} {sp.max_loss:8.2f} {sp.credit_pct:8.1f}% {sp.pop:5.2f} {ivs:>7} {srcs:>9} {sp.days_to_expiry:4d}"
                )
            print()


if __name__ == "__main__":
    main()

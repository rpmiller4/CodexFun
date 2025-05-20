from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
from typing import List, Optional

import yfinance as yf
try:
    from colorama import Fore, Style, init as colorama_init
except Exception:  # pragma: no cover - colorama optional
    class _Dummy:
        def __getattr__(self, name):
            return ""

    Fore = Style = _Dummy()
    def colorama_init():
        pass

import option_analysis as oa
from spread_analysis import SpreadType, get_credit_spreads, filter_credit_spreads
import expiry_selector
from utils import MIN_SIGMA, fetch_with_retry


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
    parser.add_argument("--widths", nargs="+", type=float, default=[2.0, 3.0, 5.0])
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
        "--sigma-floor",
        type=float,
        default=MIN_SIGMA,
        help="Override minimum volatility used when deriving VIX sigma",
    )
    parser.add_argument(
        "--show-sigma",
        action="store_true",
        help="Display sigma column in results",
    )
    parser.add_argument(
        "--log-file",
        help="Write TSV run log to this file instead of logs/run-YYYYMMDD-HHMM.tsv",
    )
    parsed = parser.parse_args(args)
    colorama_init()

    spread_type = SpreadType(parsed.type)
    ticker = yf.Ticker("SPY")
    explicit_list = parsed.expiry_dates if parsed.expiry_dates else []
    expiries = (
        explicit_list
        if explicit_list
        else fetch_with_retry(expiry_selector.expiries_within, ticker, max_days=parsed.max_days)
    )
    today = dt.date.today()

    # run log
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = "logs"
    log_path = parsed.log_file or f"{log_dir}/run-{ts}.tsv"
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        pass
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    try:
        with open(log_path, "a") as f:
            f.write(
                f"{ts}\t{' '.join(args or [])}\t{','.join(expiries)}\t{parsed.sigma_floor}\t{sha}\n"
            )
    except Exception:
        pass

    for expiry in expiries:
        for width in parsed.widths:
            spreads = get_credit_spreads(
                expiry,
                width=width,
                spread_type=spread_type,
                min_iv=parsed.min_iv,
                sigma_floor=parsed.sigma_floor,
            )
            spreads = filter_credit_spreads(
                spreads, pop_min=parsed.pop, credit_min_pct=parsed.credit_pct
            )
            if not spreads:
                print(f"--- No candidates for ${width}-wide ---")
                continue
            days = spreads[0].days_to_expiry
            stype = spread_type.value
            print(
                f"=== {stype.title()} Spreads ~{days} Days | Width ${width} ==="
            )
            header = (
                f"{'Short':>5} {'Long':>5} {'Credit':>7} {'MaxLoss':>8} {'Credit%':>8} {'PoP':>5}"
            )
            if parsed.show_sigma:
                header += f" {'σ':>4}"
            header += f" {'EV':>7} {'IVs':>7} {'Src':>9} {'Days':>4}"
            print(header)
            for sp in spreads:
                ivs = f"{sp.short.iv:.2f}/{sp.long.iv:.2f}"
                srcs = f"{sp.iv_short_src.value}/{sp.iv_long_src.value}"
                sigma_txt = f"{sp.sigma_used:.2f}"
                if sp.sigma_used < 0.10:
                    sigma_txt = f"{Fore.YELLOW}{sigma_txt}{Style.RESET_ALL}"
                row = (
                    f"{sp.short.strike:5.0f} {sp.long.strike:5.0f} {sp.credit:7.2f} {sp.max_loss:8.2f} {sp.credit_pct:8.1f}% {sp.pop:5.2f}"
                )
                if parsed.show_sigma:
                    row += f" {sigma_txt:>4}"
                row += (
                    f" {sp.expected_value:7.2f} {ivs:>7} {srcs:>9} {sp.days_to_expiry:4d}"
                )
                print(row)
            print()


if __name__ == "__main__":
    main()

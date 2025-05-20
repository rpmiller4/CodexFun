# CodexFun

Experimenting with Codex on Options trading analysis and possibly backtesting.

## SPY Options Helper

The `spy_options.py` script fetches the latest SPY price and retrieves the
options chain from Yahoo Finance. It then selects the option contract
nearest to the current price with the highest open interest for the chosen
expiry date.

Running the script requires `pandas` and `yfinance` installed as well as an
active internet connection.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python spy_options.py
```

By default the script chooses the closest expiry and prints the best call
option based on the simple heuristic above.

## Credit-Spread Screener

The `cli.py` script screens SPY bull-put or bear-call credit spreads.  Provide one or more widths and minimum IV threshold:

```bash
python cli.py --type bull_put --widths 2 5 10 --pop 0.7 --credit 25 --min-iv 0.05
```

Output columns:

| Column   | Meaning                                   |
|----------|-------------------------------------------|
| Short    | Short strike price                        |
| Long     | Long strike price                         |
| Credit   | Net credit received                       |
| MaxLoss  | Maximum possible loss                     |
| Credit%  | Credit as percentage of spread width      |
| PoP      | Theoretical probability of profit         |
| IVs      | Short/long implied volatility values      |
| Src      | IV sources for short/long legs            |
| Days     | Calendar days to expiry                   |

### IV handling

Missing or too-low implied volatility is replaced in stages:

| Source tag | Meaning |
|------------|---------|
| `orig` | IV came directly from Yahoo option chain |
| `atm`  | Missing IV replaced by at-the-money IV for that expiry |
| `vix`  | Still missing → replaced by VIX-based σ scaled to expiry |

Use `--min-iv` to change the minimum acceptable IV (default `0.05`).


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

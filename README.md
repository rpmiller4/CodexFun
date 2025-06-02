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

### Flags

| Flag | Description |
|------|-------------|
| `--widths` | Spread widths to evaluate (default `2 3 5`) |
| `--pop` | Minimum probability of profit |
| `--credit` | Minimum credit percent of width |
| `--min-iv` | Minimum acceptable implied volatility |
| `--sigma-floor` | Floor for VIX-derived volatility (default 0.12) |
| `--show-sigma` | Display σ column in output |
| `--log-file` | Path for run log TSV |

Output columns:

| Column   | Meaning                                   |
|----------|-------------------------------------------|
| Short    | Short strike price                        |
| Long     | Long strike price                         |
| Credit   | Net credit received                       |
| MaxLoss  | Maximum possible loss                     |
| Credit%  | Credit as percentage of spread width      |
| PoP      | Theoretical probability of profit         |
| σ        | Volatility used for PoP calculation       |
| EV       | Expected value of the spread              |
| IVs      | Short/long implied volatility values      |
| Src      | IV sources for short/long legs            |
| Days     | Calendar days to expiry                   |

Values in the σ column are highlighted yellow when below 10%. The EV column
uses the formula:

```
win = credit * pop
loss = (width - credit) * (1 - pop)
EV = win - loss
```

### IV handling

Missing or too-low implied volatility is replaced in stages:

| Source tag | Meaning |
|------------|---------|
| `orig` | IV came directly from Yahoo option chain |
| `atm`  | Missing IV replaced by at-the-money IV for that expiry |
| `vix`  | Still missing → replaced by VIX-based σ scaled to expiry |

Use `--min-iv` to change the minimum acceptable IV (default `0.05`).
Use `--sigma-floor` to override the hard minimum volatility applied when
falling back to the VIX (default `0.12`).

### Expiry selection
By default the screener scans *every* listed SPY expiry within the next **14 calendar days**:

```bash
python cli.py --type bull_put  # scans all expiries \u226414 days
python cli.py --max-days 30    # widen window to 30 days
python cli.py --expiry-dates 2025-06-05 2025-06-21  # override with explicit list
```

## Synthetic Market Simulation

The `synthetic_market` module can now generate price paths with optional
features:

- **AR(1) autocorrelation** via the `ar1` parameter.
- **Volatility clustering** controlled by `vol_cluster`.
- **Nonlinear price reaction** using the `reaction` exponent.
- **Realism flags** grouped under `realism`:
  - `slippage` – subtract random bid/ask spread from fills.
  - `liquidity` – probability of trade execution decreases with strike distance.
  - `margin` – enforce dynamic margin requirements per trade.
  - `jumps` – simulate rare jump-diffusion price moves.

Default realism configuration:

```python
{"slippage": False, "liquidity": False, "margin": False, "jumps": False}
```

The helper `quote_spread(distance)` computes the bid/ask spread used for
slippage when enabled.

Use `run_simulation_multi` to evaluate a configuration over multiple random
seeds and analyse the distribution of returns.


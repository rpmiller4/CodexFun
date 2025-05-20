import sys
import pandas as pd
from types import SimpleNamespace

class FakeTicker:
    def __init__(self, *args, **kwargs):
        import datetime as dt
        self.options = [(dt.date.today() + dt.timedelta(days=3)).strftime("%Y-%m-%d")]

    def history(self, period="1d"):
        return pd.DataFrame({"Close": [100.0]})

    def option_chain(self, expiry):
        df = pd.DataFrame({
            "strike": [95, 93],
            "lastPrice": [1.5, 0.5],
            "impliedVolatility": [0.2, 0.2],
        })
        return SimpleNamespace(calls=df, puts=df)


def test_cli_runs(monkeypatch, capsys):
    from cli import main
    monkeypatch.setattr("yfinance.Ticker", lambda *a, **k: FakeTicker())
    main(["--type", "bull_put", "--max-days", "3", "--widths", "2", "--show-sigma"])
    captured = capsys.readouterr()
    assert "Spreads" in captured.out

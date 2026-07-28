"""Positional breakout scanner: 20D range break + volume surge (monthly options)."""

from __future__ import annotations

import pandas as pd

from scanners.positional_common import build_positional_frame, build_trade_plan, expiry_ok

SCANNER_NAME = "Breakout"


def scan(df: pd.DataFrame) -> pd.DataFrame:
    df = expiry_ok(df)
    rows = []

    # Close breaks prior 20D high (use previous day's 20D high ≈ High20 of prior bar approx:
    # High20 includes today; require Close >= High20 * 0.998 and Close > PrevClose)
    bull = df[
        (df["Close"] >= df["High20"] * 0.999)
        & (df["Close"] > df["PrevClose"])
        & (df["Close"] > df["EMA20"])
        & (df["VolRatio"] >= 1.5)
        & (df["RSI"] >= 55)
        & (df["RSI"] <= 75)
        & (df["ExtATR"] <= 3.0)
    ]
    for _, r in bull.iterrows():
        reason = (
            f"Bullish breakout: close {r['Close']:.2f} at/through 20D high {r['High20']:.2f} "
            f"with volume {r['VolRatio']:.1f}× avg; RSI {r['RSI']:.1f}; holding above EMA20. "
            f"Momentum continuation setup for monthly CE into expiry {r['Expiry']}."
        )
        rows.append(build_trade_plan(r, "CALL", reason))

    bear = df[
        (df["Close"] <= df["Low20"] * 1.001)
        & (df["Close"] < df["PrevClose"])
        & (df["Close"] < df["EMA20"])
        & (df["VolRatio"] >= 1.5)
        & (df["RSI"] <= 45)
        & (df["RSI"] >= 25)
        & (df["ExtATR"] <= 3.0)
    ]
    for _, r in bear.iterrows():
        reason = (
            f"Bearish breakdown: close {r['Close']:.2f} at/through 20D low {r['Low20']:.2f} "
            f"with volume {r['VolRatio']:.1f}× avg; RSI {r['RSI']:.1f}; below EMA20. "
            f"Downside continuation setup for monthly PE into expiry {r['Expiry']}."
        )
        rows.append(build_trade_plan(r, "PUT", reason))

    return pd.DataFrame(rows)


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = df if df is not None else build_positional_frame()
    return scan(frame)


if __name__ == "__main__":
    hits = run()
    print(hits.head(20).to_string(index=False) if not hits.empty else "no hits")

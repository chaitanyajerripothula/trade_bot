"""Positional trend scanner: EMA stack + RSI swing zone + volume (monthly options)."""

from __future__ import annotations

import pandas as pd

from scanners.positional_common import build_positional_frame, build_trade_plan, expiry_ok

SCANNER_NAME = "Trend"


def scan(df: pd.DataFrame) -> pd.DataFrame:
    df = expiry_ok(df)
    rows = []

    # Bullish trend → CALL
    bull = df[
        (df["Close"] > df["EMA20"])
        & (df["EMA20"] > df["EMA50"])
        & (df["Close"] > df["EMA200"])
        & (df["RSI"] >= 50)
        & (df["RSI"] <= 68)
        & (df["VolRatio"] >= 1.1)
        & (df["ExtATR"] <= 2.5)
    ]
    for _, r in bull.iterrows():
        reason = (
            f"Bullish EMA stack (Close>{r['EMA20']:.1f}>{r['EMA50']:.1f}, above EMA200); "
            f"RSI {r['RSI']:.1f} in swing buy zone 50–68; volume {r['VolRatio']:.1f}× 20D avg; "
            f"not overextended ({r['ExtATR']:.1f} ATR from EMA20). Suited for monthly CE swing."
        )
        rows.append(build_trade_plan(r, "CALL", reason))

    # Bearish trend → PUT
    bear = df[
        (df["Close"] < df["EMA20"])
        & (df["EMA20"] < df["EMA50"])
        & (df["Close"] < df["EMA200"])
        & (df["RSI"] <= 50)
        & (df["RSI"] >= 32)
        & (df["VolRatio"] >= 1.1)
        & (df["ExtATR"] <= 2.5)
    ]
    for _, r in bear.iterrows():
        reason = (
            f"Bearish EMA stack (Close<{r['EMA20']:.1f}<{r['EMA50']:.1f}, below EMA200); "
            f"RSI {r['RSI']:.1f} in swing sell zone 32–50; volume {r['VolRatio']:.1f}× 20D avg; "
            f"not overextended ({r['ExtATR']:.1f} ATR from EMA20). Suited for monthly PE swing."
        )
        rows.append(build_trade_plan(r, "PUT", reason))

    return pd.DataFrame(rows)


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = df if df is not None else build_positional_frame()
    return scan(frame)


if __name__ == "__main__":
    hits = run()
    print(hits.head(20).to_string(index=False) if not hits.empty else "no hits")

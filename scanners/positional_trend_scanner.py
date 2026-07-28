"""Positional trend scanner: home-run EMA structure (monthly options)."""

from __future__ import annotations

import pandas as pd

from scanners.positional_common import (
    MIN_ATR_PCT,
    build_positional_frame,
    build_trade_plan,
    expiry_ok,
)

SCANNER_NAME = "Trend"


def scan(df: pd.DataFrame) -> pd.DataFrame:
    df = expiry_ok(df)
    atr_pct = df["ATR"] / df["Close"] * 100
    rows = []

    bull = df[
        (df["Close"] > df["EMA20"])
        & (df["EMA20"] > df["EMA50"])
        & (df["Close"] > df["EMA200"])
        & (df["RSI"] >= 52)
        & (df["RSI"] <= 68)
        & (df["VolRatio"] >= 1.3)
        & (df["ExtATR"] <= 2.2)
        & (atr_pct >= MIN_ATR_PCT)
    ]
    for _, r in bull.iterrows():
        reason = (
            f"HOME-RUN bull structure: Close>EMA20>EMA50 and above EMA200; "
            f"RSI {r['RSI']:.1f}; volume {r['VolRatio']:.1f}×; ATR {r['ATR']/r['Close']*100:.1f}% of price; "
            f"extension {r['ExtATR']:.1f} ATR (not chased)."
        )
        rows.append(build_trade_plan(r, "CALL", reason))

    bear = df[
        (df["Close"] < df["EMA20"])
        & (df["EMA20"] < df["EMA50"])
        & (df["Close"] < df["EMA200"])
        & (df["RSI"] <= 48)
        & (df["RSI"] >= 32)
        & (df["VolRatio"] >= 1.3)
        & (df["ExtATR"] <= 2.2)
        & (atr_pct >= MIN_ATR_PCT)
    ]
    for _, r in bear.iterrows():
        reason = (
            f"HOME-RUN bear structure: Close<EMA20<EMA50 and below EMA200; "
            f"RSI {r['RSI']:.1f}; volume {r['VolRatio']:.1f}×; ATR {r['ATR']/r['Close']*100:.1f}% of price; "
            f"extension {r['ExtATR']:.1f} ATR."
        )
        rows.append(build_trade_plan(r, "PUT", reason))

    return pd.DataFrame(rows)


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = df if df is not None else build_positional_frame()
    return scan(frame)


if __name__ == "__main__":
    hits = run()
    print(hits.head(20).to_string(index=False) if not hits.empty else "no hits")

"""Positional breakout scanner: home-run range break (monthly options)."""

from __future__ import annotations

import pandas as pd

from scanners.positional_common import (
    MIN_ATR_PCT,
    build_positional_frame,
    build_trade_plan,
    expiry_ok,
)

SCANNER_NAME = "Breakout"


def scan(df: pd.DataFrame) -> pd.DataFrame:
    df = expiry_ok(df)
    atr_pct = df["ATR"] / df["Close"] * 100
    rows = []

    bull = df[
        (df["Close"] >= df["High20"] * 0.999)
        & (df["Close"] > df["PrevClose"])
        & (df["Close"] > df["EMA20"])
        & (df["EMA20"] > df["EMA50"])
        & (df["VolRatio"] >= 1.8)
        & (df["RSI"] >= 55)
        & (df["RSI"] <= 72)
        & (df["ExtATR"] <= 2.8)
        & (atr_pct >= MIN_ATR_PCT)
    ]
    for _, r in bull.iterrows():
        reason = (
            f"HOME-RUN breakout through 20D high {r['High20']:.2f} on {r['VolRatio']:.1f}× volume; "
            f"EMA20>EMA50; RSI {r['RSI']:.1f}; ATR room {r['ATR']/r['Close']*100:.1f}%."
        )
        rows.append(build_trade_plan(r, "CALL", reason))

    bear = df[
        (df["Close"] <= df["Low20"] * 1.001)
        & (df["Close"] < df["PrevClose"])
        & (df["Close"] < df["EMA20"])
        & (df["EMA20"] < df["EMA50"])
        & (df["VolRatio"] >= 1.8)
        & (df["RSI"] <= 45)
        & (df["RSI"] >= 28)
        & (df["ExtATR"] <= 2.8)
        & (atr_pct >= MIN_ATR_PCT)
    ]
    for _, r in bear.iterrows():
        reason = (
            f"HOME-RUN breakdown through 20D low {r['Low20']:.2f} on {r['VolRatio']:.1f}× volume; "
            f"EMA20<EMA50; RSI {r['RSI']:.1f}; ATR room {r['ATR']/r['Close']*100:.1f}%."
        )
        rows.append(build_trade_plan(r, "PUT", reason))

    return pd.DataFrame(rows)


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = df if df is not None else build_positional_frame()
    return scan(frame)


if __name__ == "__main__":
    hits = run()
    print(hits.head(20).to_string(index=False) if not hits.empty else "no hits")

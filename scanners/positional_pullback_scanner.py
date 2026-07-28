"""Positional pullback scanner: home-run RSI reset (monthly options)."""

from __future__ import annotations

import pandas as pd

from scanners.positional_common import (
    MIN_ATR_PCT,
    build_positional_frame,
    build_trade_plan,
    expiry_ok,
)

SCANNER_NAME = "Pullback"


def scan(df: pd.DataFrame) -> pd.DataFrame:
    df = expiry_ok(df)
    atr_pct = df["ATR"] / df["Close"] * 100
    rows = []

    bull = df[
        (df["Close"] > df["EMA50"])
        & (df["EMA20"] > df["EMA50"])
        & (df["Close"] > df["EMA200"])
        & (df["RSI"] >= 44)
        & (df["RSI"] <= 56)
        & (df["RSI"] > df["RSI_5d_ago"])
        & (df["RSI_5d_ago"] <= 44)
        & (df["Close"] >= df["EMA20"] * 0.985)
        & (df["Close"] <= df["EMA20"] * 1.025)
        & (df["VolRatio"] >= 1.1)
        & (atr_pct >= MIN_ATR_PCT)
    ]
    for _, r in bull.iterrows():
        reason = (
            f"HOME-RUN bull pullback: uptrend intact; RSI {r['RSI_5d_ago']:.1f}→{r['RSI']:.1f}; "
            f"coiled at EMA20 with ATR room {r['ATR']/r['Close']*100:.1f}%."
        )
        rows.append(build_trade_plan(r, "CALL", reason))

    bear = df[
        (df["Close"] < df["EMA50"])
        & (df["EMA20"] < df["EMA50"])
        & (df["Close"] < df["EMA200"])
        & (df["RSI"] <= 56)
        & (df["RSI"] >= 44)
        & (df["RSI"] < df["RSI_5d_ago"])
        & (df["RSI_5d_ago"] >= 56)
        & (df["Close"] <= df["EMA20"] * 1.015)
        & (df["Close"] >= df["EMA20"] * 0.975)
        & (df["VolRatio"] >= 1.1)
        & (atr_pct >= MIN_ATR_PCT)
    ]
    for _, r in bear.iterrows():
        reason = (
            f"HOME-RUN bear pullback: downtrend intact; RSI {r['RSI_5d_ago']:.1f}→{r['RSI']:.1f}; "
            f"rejected near EMA20 with ATR room {r['ATR']/r['Close']*100:.1f}%."
        )
        rows.append(build_trade_plan(r, "PUT", reason))

    return pd.DataFrame(rows)


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = df if df is not None else build_positional_frame()
    return scan(frame)


if __name__ == "__main__":
    hits = run()
    print(hits.head(20).to_string(index=False) if not hits.empty else "no hits")

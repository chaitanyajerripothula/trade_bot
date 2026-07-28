"""Positional momentum scanner: home-run thrust (monthly options)."""

from __future__ import annotations

import pandas as pd

from scanners.positional_common import (
    MIN_ATR_PCT,
    build_positional_frame,
    build_trade_plan,
    expiry_ok,
)

SCANNER_NAME = "Momentum"


def scan(df: pd.DataFrame) -> pd.DataFrame:
    df = expiry_ok(df)
    atr_pct = df["ATR"] / df["Close"] * 100
    chg = (df["Close"] - df["PrevClose"]) / df["PrevClose"] * 100
    rows = []

    bull = df[
        (chg >= 1.5)
        & (df["Close"] > df["EMA20"])
        & (df["EMA20"] > df["EMA50"])
        & (df["VolRatio"] >= 1.6)
        & (df["RSI"] >= 55)
        & (df["RSI"] <= 70)
        & (df["ExtATR"] <= 2.6)
        & (atr_pct >= MIN_ATR_PCT)
    ]
    for _, r in bull.iterrows():
        day_chg = (r["Close"] - r["PrevClose"]) / r["PrevClose"] * 100
        reason = (
            f"HOME-RUN thrust +{day_chg:.1f}% on {r['VolRatio']:.1f}× volume; "
            f"EMA20>EMA50; RSI {r['RSI']:.1f}; ATR {r['ATR']/r['Close']*100:.1f}% — expansion CE."
        )
        rows.append(build_trade_plan(r, "CALL", reason))

    bear = df[
        (chg <= -1.5)
        & (df["Close"] < df["EMA20"])
        & (df["EMA20"] < df["EMA50"])
        & (df["VolRatio"] >= 1.6)
        & (df["RSI"] <= 45)
        & (df["RSI"] >= 30)
        & (df["ExtATR"] <= 2.6)
        & (atr_pct >= MIN_ATR_PCT)
    ]
    for _, r in bear.iterrows():
        day_chg = (r["Close"] - r["PrevClose"]) / r["PrevClose"] * 100
        reason = (
            f"HOME-RUN thrust {day_chg:.1f}% on {r['VolRatio']:.1f}× volume; "
            f"EMA20<EMA50; RSI {r['RSI']:.1f}; ATR {r['ATR']/r['Close']*100:.1f}% — expansion PE."
        )
        rows.append(build_trade_plan(r, "PUT", reason))

    return pd.DataFrame(rows)


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = df if df is not None else build_positional_frame()
    return scan(frame)


if __name__ == "__main__":
    hits = run()
    print(hits.head(20).to_string(index=False) if not hits.empty else "no hits")

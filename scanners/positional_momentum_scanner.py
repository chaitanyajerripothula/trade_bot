"""Positional momentum scanner: strong directional push with orderly ATR (monthly options)."""

from __future__ import annotations

import pandas as pd

from scanners.positional_common import build_positional_frame, build_trade_plan, expiry_ok

SCANNER_NAME = "Momentum"


def scan(df: pd.DataFrame) -> pd.DataFrame:
    df = expiry_ok(df)
    rows = []
    chg = (df["Close"] - df["PrevClose"]) / df["PrevClose"] * 100

    bull = df[
        (chg >= 1.0)
        & (df["Close"] > df["EMA20"])
        & (df["EMA20"] > df["EMA50"])
        & (df["VolRatio"] >= 1.3)
        & (df["RSI"] >= 55)
        & (df["RSI"] <= 72)
        & (df["ExtATR"] <= 2.8)
    ]
    for _, r in bull.iterrows():
        day_chg = (r["Close"] - r["PrevClose"]) / r["PrevClose"] * 100
        reason = (
            f"Bullish momentum day +{day_chg:.1f}% with volume {r['VolRatio']:.1f}×; "
            f"EMA20>EMA50 intact; RSI {r['RSI']:.1f} strong but not blown-off (>72 blocked). "
            f"High-beta monthly CE candidate if DTE supports swing to {r['Expiry']}."
        )
        rows.append(build_trade_plan(r, "CALL", reason))

    bear = df[
        (chg <= -1.0)
        & (df["Close"] < df["EMA20"])
        & (df["EMA20"] < df["EMA50"])
        & (df["VolRatio"] >= 1.3)
        & (df["RSI"] <= 45)
        & (df["RSI"] >= 28)
        & (df["ExtATR"] <= 2.8)
    ]
    for _, r in bear.iterrows():
        day_chg = (r["Close"] - r["PrevClose"]) / r["PrevClose"] * 100
        reason = (
            f"Bearish momentum day {day_chg:.1f}% with volume {r['VolRatio']:.1f}×; "
            f"EMA20<EMA50 intact; RSI {r['RSI']:.1f} weak but not capitulation (<28 blocked). "
            f"Monthly PE candidate for continuation into expiry {r['Expiry']}."
        )
        rows.append(build_trade_plan(r, "PUT", reason))

    return pd.DataFrame(rows)


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = df if df is not None else build_positional_frame()
    return scan(frame)


if __name__ == "__main__":
    hits = run()
    print(hits.head(20).to_string(index=False) if not hits.empty else "no hits")

"""Positional pullback scanner: trend intact + RSI reset (monthly options)."""

from __future__ import annotations

import pandas as pd

from scanners.positional_common import build_positional_frame, build_trade_plan, expiry_ok

SCANNER_NAME = "Pullback"


def scan(df: pd.DataFrame) -> pd.DataFrame:
    df = expiry_ok(df)
    rows = []

    # Bull pullback: still above EMA50/200, RSI was weaker 5d ago and recovering toward 45–58
    bull = df[
        (df["Close"] > df["EMA50"])
        & (df["EMA20"] > df["EMA50"])
        & (df["Close"] > df["EMA200"])
        & (df["RSI"] >= 42)
        & (df["RSI"] <= 58)
        & (df["RSI"] > df["RSI_5d_ago"])
        & (df["RSI_5d_ago"] <= 45)
        & (df["Close"] >= df["EMA20"] * 0.985)
        & (df["VolRatio"] >= 0.9)
    ]
    for _, r in bull.iterrows():
        reason = (
            f"Bullish pullback in uptrend: EMA20>EMA50 and price above EMA200; "
            f"RSI reset from {r['RSI_5d_ago']:.1f} → {r['RSI']:.1f} (recovering); "
            f"price near EMA20 support. Prefer monthly CE on continuation, not chase."
        )
        rows.append(build_trade_plan(r, "CALL", reason))

    bear = df[
        (df["Close"] < df["EMA50"])
        & (df["EMA20"] < df["EMA50"])
        & (df["Close"] < df["EMA200"])
        & (df["RSI"] <= 58)
        & (df["RSI"] >= 42)
        & (df["RSI"] < df["RSI_5d_ago"])
        & (df["RSI_5d_ago"] >= 55)
        & (df["Close"] <= df["EMA20"] * 1.015)
        & (df["VolRatio"] >= 0.9)
    ]
    for _, r in bear.iterrows():
        reason = (
            f"Bearish pullback in downtrend: EMA20<EMA50 and price below EMA200; "
            f"RSI faded from {r['RSI_5d_ago']:.1f} → {r['RSI']:.1f}; "
            f"price near EMA20 resistance. Prefer monthly PE on failure of bounce."
        )
        rows.append(build_trade_plan(r, "PUT", reason))

    return pd.DataFrame(rows)


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = df if df is not None else build_positional_frame()
    return scan(frame)


if __name__ == "__main__":
    hits = run()
    print(hits.head(20).to_string(index=False) if not hits.empty else "no hits")

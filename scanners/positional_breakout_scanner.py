"""Positional breakout scanner: prior-range break for 1:20 monthly options."""

from __future__ import annotations

import pandas as pd

from scanners.positional_common import (
    MIN_ATR_PCT,
    MIN_VOL_RATIO,
    build_positional_frame,
    build_trade_plan,
    expiry_ok,
)

SCANNER_NAME = "Breakout"


def scan(df: pd.DataFrame) -> pd.DataFrame:
    """
    True breakout vs PRIOR 20D extremes (PrevHigh20 / PrevLow20).

    Using High20/Low20 includes today's bar and creates look-ahead bias:
    a close that made the day's high almost always "breaks" High20.
    """
    df = expiry_ok(df)
    atr_pct = df["ATR"] / df["Close"] * 100
    rows = []

    bull = df[
        (df["Close"] > df["PrevHigh20"])
        & (df["Close"] > df["PrevClose"])
        & (df["Close"] > df["EMA20"])
        & (df["EMA20"] > df["EMA50"])
        & (df["VolRatio"] >= max(1.8, MIN_VOL_RATIO))
        & (df["RSI"] >= 55)
        & (df["RSI"] <= 72)
        & (df["ExtATR"] <= 2.8)
        & (atr_pct >= MIN_ATR_PCT)
    ]
    for _, r in bull.iterrows():
        reason = (
            f"1:20 breakout through prior 20D high {r['PrevHigh20']:.2f} "
            f"on {r['VolRatio']:.1f}× volume; EMA20>EMA50; RSI {r['RSI']:.1f}; "
            f"ATR room {r['ATR']/r['Close']*100:.1f}%."
        )
        plan = build_trade_plan(r, "CALL", reason)
        if plan:
            rows.append(plan)

    bear = df[
        (df["Close"] < df["PrevLow20"])
        & (df["Close"] < df["PrevClose"])
        & (df["Close"] < df["EMA20"])
        & (df["EMA20"] < df["EMA50"])
        & (df["VolRatio"] >= max(1.8, MIN_VOL_RATIO))
        & (df["RSI"] <= 45)
        & (df["RSI"] >= 28)
        & (df["ExtATR"] <= 2.8)
        & (atr_pct >= MIN_ATR_PCT)
    ]
    for _, r in bear.iterrows():
        reason = (
            f"1:20 breakdown through prior 20D low {r['PrevLow20']:.2f} "
            f"on {r['VolRatio']:.1f}× volume; EMA20<EMA50; RSI {r['RSI']:.1f}; "
            f"ATR room {r['ATR']/r['Close']*100:.1f}%."
        )
        plan = build_trade_plan(r, "PUT", reason)
        if plan:
            rows.append(plan)

    return pd.DataFrame(rows)


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = df if df is not None else build_positional_frame()
    return scan(frame)


if __name__ == "__main__":
    hits = run()
    print(hits.head(20).to_string(index=False) if not hits.empty else "no hits")

"""Institutional footprint scanner: matched pre-open volume vs 20-day ADV."""

from __future__ import annotations

import pandas as pd

from scanners._common import build_preopen_frame, price_bias

SCANNER_NAME = "Footprint"
FOOTPRINT_PCT_MIN = 5.0
COLUMNS = ["Symbol", "Bias", "IEP_Open", "Pct_Chg", "Matched_Vol", "20Day_ADV", "Footprint_Pct"]


def scan(df: pd.DataFrame, *, min_footprint_pct: float = FOOTPRINT_PCT_MIN) -> pd.DataFrame:
    hits = df[df["Footprint_Pct"] >= min_footprint_pct].copy()
    hits["Bias"] = hits["Pct_Chg"].map(price_bias)
    hits = hits[hits["Bias"].isin(["BUY", "SELL"])]
    return hits.sort_values(by="Footprint_Pct", ascending=False)


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = df if df is not None else build_preopen_frame()
    return scan(frame)


if __name__ == "__main__":
    hits = run()
    print(hits[COLUMNS].head(20).to_string(index=False) if not hits.empty else "no hits")

"""Institutional footprint scanner: matched pre-open volume vs 20-day ADV."""

from __future__ import annotations

import pandas as pd

from scanners._common import build_preopen_frame, emit_hits

SCANNER_NAME = "Footprint Scanner"
SUBJECT = "SCANNER: Footprint >= 5% of 20-Day ADV"
FOOTPRINT_PCT_MIN = 5.0
COLUMNS = ["Symbol", "IEP_Open", "Pct_Chg", "Matched_Vol", "20Day_ADV", "Footprint_Pct"]


def scan(df: pd.DataFrame, *, min_footprint_pct: float = FOOTPRINT_PCT_MIN) -> pd.DataFrame:
    hits = df[df["Footprint_Pct"] >= min_footprint_pct].sort_values(
        by="Footprint_Pct", ascending=False
    )
    return hits


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = df if df is not None else build_preopen_frame()
    hits = scan(frame)
    return emit_hits(SCANNER_NAME, SUBJECT, hits, COLUMNS)


if __name__ == "__main__":
    run()

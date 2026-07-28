"""Volume shock scanner: absolute matched pre-open volume vs ADV floor."""

from __future__ import annotations

import pandas as pd

from scanners._common import build_preopen_frame, emit_hits

SCANNER_NAME = "Volume Shock Scanner"
SUBJECT = "SCANNER: Pre-Open Volume Shock Matched_Vol >= 2% ADV and >= 50k"
FOOTPRINT_PCT_MIN = 2.0
MIN_MATCHED_VOL = 50_000
COLUMNS = ["Symbol", "IEP_Open", "Pct_Chg", "Matched_Vol", "20Day_ADV", "Footprint_Pct"]


def scan(
    df: pd.DataFrame,
    *,
    min_footprint_pct: float = FOOTPRINT_PCT_MIN,
    min_matched_vol: float = MIN_MATCHED_VOL,
) -> pd.DataFrame:
    hits = df[
        (df["Footprint_Pct"] >= min_footprint_pct) & (df["Matched_Vol"] >= min_matched_vol)
    ].sort_values(by="Matched_Vol", ascending=False)
    return hits


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = df if df is not None else build_preopen_frame()
    hits = scan(frame)
    return emit_hits(SCANNER_NAME, SUBJECT, hits, COLUMNS)


if __name__ == "__main__":
    run()

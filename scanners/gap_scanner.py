"""Gap scanner: large IEP % change vs previous close in pre-open."""

from __future__ import annotations

import pandas as pd

from scanners._common import build_preopen_frame, emit_hits

SCANNER_NAME = "Gap Scanner"
SUBJECT = "SCANNER: Pre-Open Gap |Pct_Chg| >= 1.5%"
GAP_PCT_MIN = 1.5
COLUMNS = ["Symbol", "IEP_Open", "Prev_Close", "Gap_Pct", "Matched_Vol", "Footprint_Pct"]


def scan(df: pd.DataFrame, *, min_abs_gap_pct: float = GAP_PCT_MIN) -> pd.DataFrame:
    hits = df[df["Gap_Pct"].abs() >= min_abs_gap_pct].copy()
    hits["Gap_Direction"] = hits["Gap_Pct"].apply(lambda x: "UP" if x >= 0 else "DOWN")
    return hits.sort_values(by="Gap_Pct", key=lambda s: s.abs(), ascending=False)


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = df if df is not None else build_preopen_frame()
    hits = scan(frame)
    cols = COLUMNS + ["Gap_Direction"]
    return emit_hits(SCANNER_NAME, SUBJECT, hits, cols)


if __name__ == "__main__":
    run()

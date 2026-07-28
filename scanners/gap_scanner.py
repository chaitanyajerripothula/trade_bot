"""Gap scanner: large IEP % change vs previous close in pre-open."""

from __future__ import annotations

import pandas as pd

from scanners._common import build_preopen_frame

SCANNER_NAME = "Gap"
# High-conviction gap floor (was 1.5% for noisy alerts).
GAP_PCT_MIN = 2.0
COLUMNS = ["Symbol", "Bias", "IEP_Open", "Prev_Close", "Gap_Pct", "Matched_Vol", "Footprint_Pct"]


def scan(df: pd.DataFrame, *, min_abs_gap_pct: float = GAP_PCT_MIN) -> pd.DataFrame:
    hits = df[df["Gap_Pct"].abs() >= min_abs_gap_pct].copy()
    hits["Bias"] = hits["Gap_Pct"].apply(lambda x: "BUY" if x > 0 else "SELL")
    return hits.sort_values(by="Gap_Pct", key=lambda s: s.abs(), ascending=False)


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = df if df is not None else build_preopen_frame()
    return scan(frame)


if __name__ == "__main__":
    hits = run()
    print(hits[COLUMNS].head(20).to_string(index=False) if not hits.empty else "no hits")

"""52-week proximity scanner: IEP near year high or year low."""

from __future__ import annotations

import pandas as pd

from scanners._common import build_preopen_frame, emit_hits

SCANNER_NAME = "52W Proximity Scanner"
SUBJECT = "SCANNER: IEP within 3% of 52W High or Low"
PROXIMITY_PCT = 3.0
COLUMNS = [
    "Symbol",
    "IEP_Open",
    "Year_High",
    "Year_Low",
    "Pct_From_52W_High",
    "Pct_From_52W_Low",
    "Pct_Chg",
    "Matched_Vol",
]


def scan(df: pd.DataFrame, *, proximity_pct: float = PROXIMITY_PCT) -> pd.DataFrame:
    near_high = df["Pct_From_52W_High"].abs() <= proximity_pct
    near_low = df["Pct_From_52W_Low"].abs() <= proximity_pct
    hits = df[(near_high | near_low) & (df["Year_High"] > 0) & (df["Year_Low"] > 0)].copy()
    hits["Proximity"] = hits.apply(
        lambda r: "NEAR_HIGH"
        if abs(r["Pct_From_52W_High"]) <= abs(r["Pct_From_52W_Low"])
        else "NEAR_LOW",
        axis=1,
    )
    return hits.sort_values(by="Matched_Vol", ascending=False)


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = df if df is not None else build_preopen_frame()
    hits = scan(frame)
    cols = COLUMNS + ["Proximity"]
    return emit_hits(SCANNER_NAME, SUBJECT, hits, cols)


if __name__ == "__main__":
    run()

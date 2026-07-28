"""52-week proximity scanner: IEP near year high or year low."""

from __future__ import annotations

import pandas as pd

from scanners._common import build_preopen_frame

SCANNER_NAME = "Near52W"
PROXIMITY_PCT = 3.0
COLUMNS = [
    "Symbol",
    "Bias",
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
    hits["Bias"] = hits.apply(
        lambda r: "BUY"
        if abs(r["Pct_From_52W_High"]) <= abs(r["Pct_From_52W_Low"])
        else "SELL",
        axis=1,
    )
    # High-conviction: bias should agree with pre-open price direction when move is meaningful.
    aligned = hits[
        ((hits["Bias"] == "BUY") & (hits["Pct_Chg"] >= 0))
        | ((hits["Bias"] == "SELL") & (hits["Pct_Chg"] <= 0))
    ]
    return aligned.sort_values(by="Matched_Vol", ascending=False)


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = df if df is not None else build_preopen_frame()
    return scan(frame)


if __name__ == "__main__":
    hits = run()
    print(hits[COLUMNS].head(20).to_string(index=False) if not hits.empty else "no hits")

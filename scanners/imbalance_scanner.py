"""Order-book imbalance scanner: pre-open buy vs sell quantity skew."""

from __future__ import annotations

import pandas as pd

from scanners._common import build_preopen_frame, emit_hits

SCANNER_NAME = "Imbalance Scanner"
SUBJECT = "SCANNER: Pre-Open Buy/Sell Imbalance |Imbalance| >= 55%"
IMBALANCE_PCT_MIN = 55.0
MIN_BOOK_QTY = 20_000
COLUMNS = [
    "Symbol",
    "IEP_Open",
    "Pct_Chg",
    "Buy_Qty",
    "Sell_Qty",
    "Imbalance_Pct",
    "Matched_Vol",
]


def scan(
    df: pd.DataFrame,
    *,
    min_abs_imbalance_pct: float = IMBALANCE_PCT_MIN,
    min_book_qty: float = MIN_BOOK_QTY,
) -> pd.DataFrame:
    book = df["Buy_Qty"] + df["Sell_Qty"]
    hits = df[
        (df["Imbalance_Pct"].abs() >= min_abs_imbalance_pct) & (book >= min_book_qty)
    ].copy()
    hits["Bias"] = hits["Imbalance_Pct"].apply(lambda x: "BUY" if x >= 0 else "SELL")
    return hits.sort_values(by="Imbalance_Pct", key=lambda s: s.abs(), ascending=False)


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = df if df is not None else build_preopen_frame()
    hits = scan(frame)
    cols = COLUMNS + ["Bias"]
    return emit_hits(SCANNER_NAME, SUBJECT, hits, cols)


if __name__ == "__main__":
    run()

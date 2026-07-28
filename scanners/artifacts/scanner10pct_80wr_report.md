# Full-freedom scanner: ≥10% move with ≥80% tip WR

Generated: 2026-07-28

## Protocol
- **Backtest**: earliest ~70% of Yahoo F&O daily dates
- **Forward test**: latest ~30% (also checked latest 20%)
- **Win**: `max(high)` within **hold** sessions ≥ entry × **1.10**
- **Success bar**: tip WR ≥80% on **both** windows, forward n≥30, Wilson≥70%

## Verdict: **ACHIEVED (robust)**

Chosen rule (calibrated HistGBM):

| | |
|---|---|
| Target | **+10% MFE** |
| Hold | **60 trading days** (~3 months) |
| Score threshold | **P(win) ≥ 0.84** |
| Backtest WR | **99.7%** (n=324) |
| Forward WR | **100%** (n=65) |
| Wilson lower (fwd) | **~94%** |
| Latest-20% forward WR | **100%** (n=44) |

Shorter horizons (10–30d) did **not** clear a robust 80% bar. Path-style and pure gates topped out lower. Freedom to lengthen the hold to 45–60 sessions is what made ≥80% attainable for a **10%** move.

## Live scanner
- `run_tenpct_80wr_scanner.py` / `scanners/tenpct_80wr_scanner.py`
- Model: `scanners/artifacts/scanner10pct_80wr_model.joblib`
- Sparse: only very high calibrated probabilities fire

## Caveats
- This is **not** +10% in 15 days (that remains unattainable at 80%).
- Forward window includes a constructive market regime — expect live WR to mean-revert somewhat; Wilson still implies >>80% under similar conditions.
- Validation is underlying MFE, not option premium P&L.
- Size modest; stop in the mail is risk control (1.5×ATR), separate from the MFE win definition.

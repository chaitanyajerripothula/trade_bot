# TwentyPct80 Daily — ≥1 stock/day, ≥20% move, ≥80% tip WR

Generated: 2026-07-28

## Constraint
User requirement: **at least 1 stock tip per trading day**, while keeping **+20% MFE** and **≥80% tip WR** on backtest + forward.

## Tradeoff (why the old sparse rule fails this)
| Rule | Tips/day | Forward WR | Keeps 80%? |
|---|---:|---:|---|
| Old sparse thr≈0.79, hold=60 | ~0.15 | ~100% | yes, but **not 1/day** |
| thr≈0.73, hold=60 | ~1.0 | **~68%** | **no** |
| thr≈0.88, hold=180 | ~1.1 | ~83% | yes |
| **daily top-1, hold=180** | **1.0** | **~88.7%** | **yes** |
| daily top-1, hold=252 | 1.0 | ~95.3% | yes (thinner labeled fwd n=43) |

## Chosen: daily top-1 @ hold=180
Every session tip the single highest calibrated `P(+20% MFE within 180 sessions)`.

| | Backtest | Forward | Latest 20% labeled |
|---|---|---|---|
| Tip WR | **99.3%** | **88.7%** (Wilson ~82%) | **91.9%** |
| Tips/day | **1.0** | **1.0** | **1.0** |
| n | 688 | 115 | 161 |

Hold ≈ **9 months** of trading sessions. That length is the cost of keeping both **1/day** and **≥80%** for a true +20% target.

## Live
- `run_twenty_pct_80wr_scanner.py` — always emits top-1
- Model: `scanners/artifacts/scanner20pct_80wr_model.joblib` (`mode=daily_topk`, `k=1`, `hold=180`)
- Study: `scanners/artifacts/scanner20pct_1perday_study.json`

## Caveats
- Underlying MFE, not option P&L.
- Stop in email (1.5×ATR) is risk control, separate from win label.
- If you insist on ≤60d **and** 1/day, expect ~68% forward WR — not 80%.

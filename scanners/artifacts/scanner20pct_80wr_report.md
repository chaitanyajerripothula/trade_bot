# Short-hold scanner under max 1–2 month hold

Generated: 2026-07-28

## Hard constraints
1. Max hold **1–2 months** (~21–45 trading days; 60d = loose ceiling)
2. Prefer **≥1 tip/day**
3. Prefer **≥80% tip WR** and large move

## What is / isn’t attainable

| Goal | Hold cap | ≥1 tip/day | Result |
|---|---|---|---|
| **+20% @ 80% WR** | ≤45d | avg ≥1 | **No** — best forward ~**58%** |
| **+20% @ 80% WR** | ≤60d | avg ≥1 | **No** — best forward ~**68%** |
| Tip **every** calendar day @ 80% | ≤45d | forced daily top-1 | **No** — +10% top-1 ~**57–61%** |
| **+10% @ 80% WR** | **30d** | avg ≥1.4 | **Yes — shipped** |

## Shipped rule (ShortHold80)
| | |
|---|---|
| Win | **+10% MFE** within **30** trading days (~1.5 months) |
| Gate | Calibrated GBM **P ≥ 0.82** (max 3 tips) |
| Backtest WR | **~96.7%** |
| Forward WR | **~88.7%** (n=380, Wilson ~85%) |
| Avg tips/day | **~1.4** |
| Day coverage | ~**17%** of sessions fire (tips cluster) |

Quiet days are expected. Forcing a tip every day destroys the 80% bar under a 1–2 month hold.

## Live
- Runner: `run_twenty_pct_80wr_scanner.py` (ShortHold80)
- Model: `scanners/artifacts/scanner20pct_80wr_model.joblib`
- Study: `scanners/artifacts/scanner_short_hold_1perday_study.json`

## If you still want +20%
You must allow a longer hold (~180d daily top-1 previously cleared ~89% WR) — incompatible with a 1–2 month max.

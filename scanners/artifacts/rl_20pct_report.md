# RL research — ±15% in 15–30 trading days

Generated: 2026-07-28T10:12:36.400140+00:00

## Your observation is correct
F&O names **do** print ≥15% swings inside a month all the time:
- Unconditional P(+15% MFE within 30 sessions) ≈ **23.0%** of day-rows
- So in a 200-name universe, **many** names (not just one) typically manage it each month

That is **occurrence**. A scanner win-rate is different:
> Of the names we alert, what fraction actually deliver ±15% in ≤30d?

## Walk-forward prediction (Yahoo 5y, 210 names)
- CALL base rate (test): see live thresholds below
- CALL OOS AUC: **0.686**
- Best CALL precision: **57.1%** (n=21, thr=0.60)
- Usable CALL live thr: **45.1%** (n=1367, thr=0.45)
- PUT best precision: **23.2%** (n=1558)

**≥80% tip win-rate for ±15%/30d: not attained** with technical features + calibrated GBM/Q-learning on this universe.

## Live scanner
Uses max-precision calibrated thresholds (not an 80% claim) via `run_rl_20pct_scanner.py`.

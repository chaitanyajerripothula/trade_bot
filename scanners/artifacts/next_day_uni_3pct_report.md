# NextDay3 — next-day ≥3% from open

Generated: 2026-07-28T12:36:19.833827+00:00

## Objective
Next-day stocks that move **at least 3% from open in any direction**, tip WR ≥80%.

## Win definition (shipped)
`max(next_high/next_open − 1, 1 − next_low/next_open) ≥ 3%`

Tip at **EOD**; evaluates on the **next** session from its open.

## Validation
| | Backtest | Forward |
|---|---|---|
| Tip WR | **93.0%** | **85.0%** (Wilson 70.9%) |
| Tips | 457 | 40 · ~0.14/day ≈ 3/month |
| Rule | Calibrated GBM P≥0.82 (max 3) | walk-forward |

Base rate (forward) ≈ 15.5%.

## Not attainable at ≥80% tip WR
- **Strict unidirectional** (e.g. +3% one way with adverse ≤1–2%): best tip WR ~37–44%
- **Directional** CALL/PUT for which side makes the ≥3%: best ~45–48% (oracle direction recovers the 85% either-move WR)

Among either3 tips, only ~35% are strictly unidirectional on the forward window — the edge is predicting a **large open-excursion**, not a one-sided trend day.

## Live
- Model: `scanners/artifacts/next_day_3_model.joblib`
- Policy: `scanners/artifacts/next_day_3_policy.json`
- Runner: `python run_next_day_3_scanner.py --dry-run`

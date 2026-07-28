# ShortHold family — scanners with ShortHold80-like metrics

Generated: 2026-07-28

## Bar (same spirit as ShortHold80)
- Tip WR ≥80% on **backtest and forward**
- Hold ≤ **60** trading days (1–2 month budget)
- Forward n≥40, Wilson≥70%, avg tips/day ≥0.5

## Finding
- **CALL** side: many rules clear the bar across +8% / +10% / +12% / +15%
- **PUT** side: **none** cleared ≥80% forward WR under ≤60d with usable tip rate (best ~65–79%)

## Live family (shipped)

| Scanner | Target | Hold | Thr | Fwd WR | Wilson | Tips/day | ≈/month | Runner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **ShortHold80** (existing) | +10% | 30 | 0.82 | ~88.7% | ~85% | ~1.4 | 24–30 | `run_short_hold_80_scanner.py` |
| **ShortHold8** | +8% | 42 | 0.92 | **97.1%** | 94.7% | **1.25** | ~26 | `run_short_hold_8_scanner.py` |
| **ShortHold12** | +12% | 45 | 0.88 | **93.7%** | 88.5% | 0.57 | ~12 | `run_short_hold_12_scanner.py` |
| **ShortHold15** | +15% | 60 | 0.85 | **91.5%** | 85.8% | 0.60 | ~13 | `run_short_hold_15_scanner.py` |

Quiet days remain normal (tips cluster). Exit in history = planned target price.

## Research
- `research/short_hold_family_search.py`
- `scanners/artifacts/short_hold_family_study.json`
- Shared live module: `scanners/short_hold_family_scanner.py`

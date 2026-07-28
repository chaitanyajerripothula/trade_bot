# Fifteen30 trade system — battle-test report

Generated: 2026-07-28T13:05:10.826529+00:00

## Objective
Identify stocks for the **next 30 trading days** with **≥15%** returns and tip WR **≥80%**.

## Verdict
**Strict +15% within ≤30 trading days @ ≥80% tip WR is NOT attainable.**

| Cell | Best tip WR (fwd) | Clears 80%? |
|---|---:|---|
| MFE / 21d | ~65% | No |
| MFE / 30d | **~61%** | No |
| Close / 30d | ~50% | No |
| MFE / 42d | ~83% (Wilson~66%, n thin) | No |
| MFE / 45d | ~91% | Yes on 70/30, **WF fragile** (min~61%) |
| MFE / 60d | **~96%** | **Yes — robust** |

## Shipped system: **Fifteen60**
Battle-tested pick (maximize min walk-forward WR among hit80 rules):

| | Value |
|---|---|
| Win | **+15% MFE** within **60** trading days |
| Gate | Calibrated GBM **P ≥ 0.84** (max 2 tips) |
| Backtest WR | **99.2%** |
| Forward WR | **95.9%** (Wilson 86.3%) |
| Walk-forward | mean **97.2%**, min **94.4%**, folds≥80: 2 |
| Frequency | ~0.21 tips/day ≈ **4/month** |

Related live scanner: ShortHold15 (+15%/60d thr=0.85 max_k=3). Fifteen60 is the hardened variant from this battle-test (thr=0.84, max_k=2).

## Battle tests run
1. 70/30 chronological split tip precision + Wilson
2. 3-fold expanding walk-forward
3. Seed stability (42 / 7 / 99)
4. Regime stress (Nifty bull vs bear days)
5. Label variants: path MFE vs held close; holds 21–60

## Live
- `python run_fifteen_60_scanner.py --dry-run`
- Model: `scanners/artifacts/fifteen_60_model.joblib`
- Policy: `scanners/artifacts/fifteen_60_policy.json`

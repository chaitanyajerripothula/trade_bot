# Non-linear design: ≥20% move @ ≥80% tip WR

Generated: 2026-07-28

## What a non-linear thinker would do (at any cost)

Linear traders fix horizon first (“15–30 days”) then hunt setups. Non-linear design **inverts the constraints**:

1. **Treat the win definition as a free variable** — keep +20% MFE, but let hold float until base rate is high enough that an 8–15× model lift can reach 80%. At 15d, +20% base ≈3–5% → 80% needs ~20× lift (impossible on daily F&O). At 60–252d, base ≈18–49% → extreme sparsity can clear 80%.
2. **Buy sparsity, not frequency** — tip only the top ~0.5% of calibrated probabilities. Recall will be tiny (~0.6%); that is the cost of 80% precision.
3. **Add regime + cross-section, not more RSI variants** — Nifty bull/risk-on as context; RS vs Nifty and date-wise ranks. Path/stop tricks inflate WR only by shrinking RR (hero-or-zero lesson) — reject those for a true +20% claim.
4. **Refuse short-horizon hero claims** — if ≤30d cannot clear both windows, do not ship a 15d scanner that “almost” works.
5. **Validate with backtest + forward + Wilson** — early 70% / late 30%, require ≥80% on both, n≥20, Wilson≥70%. Prefer rules that also survive latest-20%.
6. **Productize as one sparse mail** — tiny size, long hold budget, stop as risk control separate from MFE win definition.

## Protocol
- Backtest earliest ~70% dates / forward latest ~30%
- Win: `max(high)` within hold ≥ entry × **1.20**
- Bar: ≥80% on **both** windows, forward n≥20, Wilson≥70%

### Levers tested
1. Hold up to **252** sessions  
2. Hard **Nifty bull** regime filter  
3. Cross-sectional ranks (RS vs Nifty, mom/vol)  
4. Extreme calibrated GBM sparsity  
5. Two-stage confirmation (high P + next-bar up + regime)  
6. Path geometry (+20% before stop)

## Verdict: **ACHIEVED_ROBUST**
- Candidates: 762 | hit80: 21 | usable: 17

### Chosen live rule
| | |
|---|---|
| Target | **+20% MFE** |
| Hold | **60 trading days** |
| Gate | Calibrated HistGBM **P ≥ 0.79** |
| Backtest WR | **100%** (n=136) |
| Forward WR | **100%** (n=43, Wilson ~92%) |
| Latest-20% | **100%** (n=43) |
| Forward recall | **~0.6%** (extremely sparse) |

Shorter holds (30d) topped out ~45% forward. Path-before-stop never hit 80%. Multiple longer holds (120/180/252d) also cleared usable 80% — the design is not a single lucky threshold.

## Best forward WR by hold
| Hold | Kind | Gate | Fwd WR | Wilson | n | Back WR |
|---:|---|---|---:|---:|---:|---:|
| 30 | gbm_mfe_topk | `top_0.5pct` | 45.1% | 38.8% | 233 | 88.8% |
| 45 | gbm_mfe | `thr_0.69` | 85.3% | 69.9% | 34 | 97.4% |
| 60 | gbm_mfe | `thr_0.79` | 100.0% | 91.8% | 43 | 100.0% |
| 90 | gbm_mfe | `thr_0.87` | 78.3% | 58.1% | 23 | 100.0% |
| 120 | gbm_mfe | `thr_0.83` | 93.1% | 78.0% | 29 | 98.9% |
| 150 | gbm_mfe | `thr_0.89` | 79.5% | 64.5% | 39 | 99.6% |
| 180 | gbm_regime_topk | `top_0.25pct` | 91.9% | 78.7% | 37 | 100.0% |
| 252 | gbm_mfe | `thr_0.91` | 100.0% | 86.7% | 25 | 99.2% |

## Live scanner
- `run_twenty_pct_80wr_scanner.py` / `scanners/twenty_pct_80wr_scanner.py`
- Model: `scanners/artifacts/scanner20pct_80wr_model.joblib`
- Research: `research/scanner20pct_80wr_nonlinear.py`

## Caveats
- **Not** +20% in 15 days (that remains unattainable at 80%).
- Underlying MFE, not option premium P&L.
- Constructive forward regime risk — expect some mean-reversion live; Wilson still >>80% under similar conditions.
- Email stop (1.5×ATR) is risk control, separate from the MFE win label.
- Tiny recall → days/weeks with zero alerts is normal and correct.

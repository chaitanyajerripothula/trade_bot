# Study: +20% movers within 15 trading days @ ≥80% tip win-rate

Generated: 2026-07-28 (Yahoo F&O walk-forward)

## Definition
- **WIN**: `max(high)` over the next **15** trading sessions ≥ entry × **1.20**
- **Tip win-rate**: fraction of *alerts* that WIN on a time-based held-out test set

## Data
- NSE F&O universe via Yahoo **5y** daily bars: **210** names, **~197k** labeled rows
- Unconditional base rate of +20% / 15d: **4.68%** (recent test window **2.64%**)
- Model: isotonic-calibrated HistGBM on trend/RSI/ATR%/volume/momentum/breakout/52w/gap features
- OOS AUC ≈ **0.76** (some signal, far from precise)

## Is ≥80% tip WR attainable?

| Method | Cleared ≥80% train & test? |
|---|---|
| Discrete technical gates | **0** |
| Calibrated GBM score thresholds (n≥12) | **0** |
| Top 1% model scores | **~21%** precision (not 80%) |
| **Verdict** | **NOT ATTAINABLE** on this data |

### Best tip precision found
| Approach | Test precision | n | Notes |
|---|---:|---:|---|
| GBM thr≈0.20 | **21.0%** | 534 | best usable threshold |
| Top 1% scores | **21.1%** | 606 | same ballpark |
| Best gate `hi_atr_trend` | **16.0%** | 188 | ATR%≥4 + bull stack |
| Base rate (test) | **2.6%** | — | random alert |

### Gate leaderboard (all << 80%)
| Gate | Test WR | n | Train WR |
|---|---:|---:|---:|
| hi_atr_trend | 16.0% | 188 | 17.8% |
| coil_release | 8.9% | 45 | 9.2% |
| extreme_vol | 8.3% | 121 | 19.5% |
| mom20_power | 6.9% | 930 | 12.7% |
| gap_cont | 6.1% | 212 | 9.9% |
| near52_thrust | 5.9% | 593 | 13.9% |
| thrust_break | 3.5% | 319 | 11.6% |
| break_vol | 3.0% | 674 | 8.8% |

## Why 80% fails here
+20% in 15 sessions is a **fat-tail** (~1-in-20 historically, rarer recently). Hitting **80% tip precision** needs roughly a **15–30×** lift over base rate. Daily technicals on F&O names only deliver about a **~8×** lift at best (~21% vs ~2.6% test base) — real edge, but not 80%.

## Practical takeaway
Do **not** build a live scanner that claims 80% for +20%/15d. Closest honest product remains:
- high-WR **small** path trades (target &lt; stop, RR≈0.5–0.75) from the hero-or-zero study, or
- sparse **lottery** home-run tickets with expected low WR.

Artifact JSON: `scanners/artifacts/move20_in_15d_study.json`

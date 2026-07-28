# +20% in 15 days @ 80% WR — without technicals

Generated: 2026-07-28T10:40:22.157392+00:00

## Question
If we **forget technicals**, can we still get **≥80% tip WR** for +20% in 15 sessions?

## What we used instead
Yahoo/public **non-price** features only:
- Sector / industry
- Market cap
- Trailing PE, margins, revenue/earnings growth
- Short % float, institutional %
- Analyst recommendation mean
- Calendar (month/weekday)
- Earnings proximity windows

No RSI, EMA, ATR, breakouts, or price momentum.

## Results

| Base rate (120-name study) | ~**5.6%** (test ~**3.3%**) |
| Best non-tech **slice** tip WR | **~8.3%** (`month=4`) |
| Non-tech GBM OOS AUC | **0.767** |
| Best non-tech GBM tip precision | **14.5%** (n=311) |
| Slices / model thresholds ≥80% | **0** |
| **Verdict** | **NOT ATTAINABLE** |

### Top slices (all << 80%)
| Slice | Test WR | n | Train WR |
|---|---:|---:|---:|
| month=4 | 8.3% | 4680 | 5.6% |
| month=3 | 7.6% | 4560 | 8.1% |
| mcap_q=0 | 6.7% | 8438 | 11.5% |
| sector=Industrials | 5.8% | 7338 | 10.7% |
| earnGrowth>30% | 5.0% | 14674 | 8.4% |

## Comparison to technicals
Earlier technical study best tip precision was ~**21%**. Non-technical public features here peak lower (~**8–15%** depending on model threshold).
Dropping technicals makes 80% **harder**, not easier.

## What *could* approach 80%
Only labels with near-certain causal payoff, e.g.:
- Announced acquisition at known premium
- Hard catalyst with historically reliable gap (still usually <<80% after costs)
- Non-public / delayed information (not usable for a live scanner)
Public sector/fundamental/calendar features do not identify +20%/15d movers at 80% precision.

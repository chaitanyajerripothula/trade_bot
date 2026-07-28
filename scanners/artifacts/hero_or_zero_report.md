# Hero-or-zero deep study

Generated: 2026-07-28T10:28:31.729371+00:00

## Question
Can we find **hero-or-zero** path trades (target before stop) with **≥80%** walk-forward win rate?

## Answer: **31** rules hit ≥80% on train+test (0 with RR≥1.5; 26 with Wilson≥70% and +EV).

| Mode | Gate | Stop | Target | Hold | RR | Test WR | Wilson | n | Train | E[R] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| resolved_only | thrust_bull | 1.5R | 0.75R | 1 | 0.5 | 95.2% | 86.9% | 63 | 88.6% | 0.429 |
| resolved_only | break_bull | 2.0R | 1.0R | 1 | 0.5 | 93.8% | 85.0% | 64 | 91.9% | 0.406 |
| resolved_only | thrust_bull | 1.5R | 1.0R | 1 | 0.67 | 93.3% | 82.1% | 45 | 85.2% | 0.556 |
| resolved_only | thrust_bull | 2.0R | 1.0R | 1 | 0.5 | 93.3% | 82.1% | 45 | 92.1% | 0.4 |
| resolved_only | gap_bull | 2.0R | 1.0R | 2 | 0.5 | 92.9% | 83.0% | 56 | 81.7% | 0.393 |
| resolved_only | break_bull | 1.5R | 0.75R | 1 | 0.5 | 92.2% | 84.8% | 90 | 88.2% | 0.383 |
| resolved_only | thrust_bull | 2.0R | 1.0R | 2 | 0.5 | 91.9% | 83.4% | 74 | 87.1% | 0.378 |
| resolved_only | gap_bull | 2.0R | 1.0R | 3 | 0.5 | 90.8% | 81.3% | 65 | 80.3% | 0.362 |
| resolved_only | coil_bull | 2.0R | 1.0R | 1 | 0.5 | 90.1% | 81.0% | 71 | 91.9% | 0.352 |
| resolved_only | thrust_bull | 2.0R | 1.0R | 3 | 0.5 | 90.1% | 82.7% | 101 | 83.2% | 0.351 |
| resolved_only | pull_bull | 2.0R | 1.0R | 3 | 0.5 | 90.0% | 69.9% | 20 | 80.8% | 0.35 |
| resolved_only | break_bull | 1.5R | 1.0R | 1 | 0.67 | 89.5% | 80.0% | 67 | 84.7% | 0.493 |
| resolved_only | squeeze_bull | 2.0R | 1.0R | 1 | 0.5 | 89.3% | 72.8% | 28 | 95.3% | 0.339 |
| resolved_only | coil_bull | 2.0R | 1.0R | 2 | 0.5 | 89.0% | 82.7% | 137 | 89.5% | 0.336 |
| resolved_only | break_bull | 2.0R | 1.0R | 2 | 0.5 | 88.9% | 81.6% | 108 | 86.0% | 0.333 |
| resolved_only | thrust_bull | 1.5R | 0.75R | 2 | 0.5 | 88.5% | 80.9% | 104 | 82.2% | 0.327 |
| resolved_only | squeeze_bull | 2.0R | 1.0R | 2 | 0.5 | 88.2% | 76.6% | 51 | 93.8% | 0.324 |
| resolved_only | break_bull | 2.0R | 1.5R | 1 | 0.75 | 87.1% | 71.2% | 31 | 82.7% | 0.524 |
| resolved_only | break_bull | 1.5R | 0.75R | 2 | 0.5 | 87.0% | 80.6% | 146 | 80.8% | 0.305 |
| resolved_only | coil_bull | 1.5R | 0.75R | 1 | 0.5 | 86.7% | 79.4% | 120 | 89.4% | 0.3 |
| resolved_only | thrust_bull | 2.0R | 1.5R | 1 | 0.75 | 86.4% | 66.7% | 22 | 83.5% | 0.511 |
| resolved_only | break_bull | 2.0R | 1.0R | 3 | 0.5 | 85.7% | 79.0% | 140 | 83.2% | 0.286 |
| resolved_only | squeeze_bull | 2.0R | 1.0R | 3 | 0.5 | 85.0% | 75.6% | 80 | 91.5% | 0.275 |
| resolved_only | pull_bull | 1.5R | 0.75R | 2 | 0.5 | 85.0% | 64.0% | 20 | 80.0% | 0.275 |
| resolved_only | coil_bull | 2.0R | 1.0R | 3 | 0.5 | 84.3% | 78.5% | 191 | 85.6% | 0.264 |
| resolved_only | near52_bull | 2.0R | 1.0R | 1 | 0.5 | 83.9% | 76.2% | 118 | 89.1% | 0.258 |
| resolved_only | squeeze_bull | 1.5R | 0.75R | 2 | 0.5 | 83.6% | 73.4% | 73 | 85.5% | 0.253 |
| resolved_only | squeeze_bull | 1.5R | 0.75R | 1 | 0.5 | 82.2% | 68.7% | 45 | 88.6% | 0.233 |
| resolved_only | coil_bull | 1.5R | 0.75R | 2 | 0.5 | 80.9% | 75.1% | 215 | 83.1% | 0.214 |
| resolved_only | squeeze_bull | 2.0R | 1.0R | 5 | 0.5 | 80.9% | 72.6% | 110 | 85.3% | 0.214 |

## Live use
Only rules in `rules_usable` (Wilson≥70%, +EV) should drive alerts.
Large-RR ‘hero’ (≫2R) almost never appears in the 80% set — 80% WR favors small targets.

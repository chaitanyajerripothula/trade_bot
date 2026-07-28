# trade_bot

NSE F&O scanner suite:
- **Morning** — pre-open high-conviction bias (one mail)
- **Evening** — 20-day ADV harvest
- **Positional** — ShortHold family + NextDay3 (≥80% tip WR)

## Pre-open scanners (`run_morning_scanners.py`)

| File | Signal |
|---|---|
| `scanners/footprint_scanner.py` | Matched pre-open vol ≥ **5%** of 20-day ADV → Bias |
| `scanners/gap_scanner.py` | \|IEP % chg\| ≥ **2%** → Bias |
| `scanners/imbalance_scanner.py` | \|Buy−Sell\| / (Buy+Sell) ≥ **55%** (book ≥ 20k) → Bias |
| `scanners/volume_shock_scanner.py` | Footprint ≥ **2%** ADV and matched vol ≥ **50k** → Bias |
| `scanners/near_52w_scanner.py` | IEP within **3%** of 52W high/low, aligned with % chg → Bias |

Morning clubs hits into **one high-conviction email** (symbols with ≥2 scanners agreeing on BUY/SELL, or strong footprint alone).

```bash
python -m scanners.gap_scanner          # one scanner (print only)
python run_morning_scanners.py          # all → one mail
```

## ShortHold family (positional)

Validated CALL scanners with ShortHold80-like metrics (hold ≤60d, tip WR ≥80%).  
PUT-side counterparts did **not** clear 80% under this hold cap.

| Scanner | Target | Hold | Fwd WR | ≈ tips/month | Runner |
|---|---:|---:|---:|---:|---|
| **ShortHold80** | +10% | 30d | ~88.7% | 24–30 | `run_short_hold_80_scanner.py` |
| **ShortHold8** | +8% | 42d | ~97.1% | ~26 | `run_short_hold_8_scanner.py` |
| **ShortHold12** | +12% | 45d | ~93.7% | ~12 | `run_short_hold_12_scanner.py` |
| **ShortHold15** | +15% | 60d | ~91.5% | ~13 | `run_short_hold_15_scanner.py` |

```bash
python run_short_hold_80_scanner.py --dry-run
python run_short_hold_8_scanner.py --dry-run
python run_short_hold_12_scanner.py --dry-run
python run_short_hold_15_scanner.py --dry-run
```

Report: `scanners/artifacts/short_hold_family_report.md`

## NextDay3 (next session ≥3% from open)

EOD tip for **tomorrow**: stock makes ≥**3%** excursion from the open in **either** direction  
(`max(open→high, open→low) ≥ 3%`). Bias=`MOVE` — direction is **not** part of the 80% edge.

| | Backtest | Forward |
|---|---|---|
| Tip WR | **~93%** | **~85%** (Wilson~71%) |
| Rule | Calibrated GBM `P≥0.82` (max 3) | walk-forward |
| Frequency | — | ~0.14 tips/day ≈ **~3/month** |

**Not attainable @ ≥80% tip WR:** strict unidirectional (adverse ≤1–2%), and directional CALL/PUT for which side makes the 3%.

```bash
python run_next_day_3_scanner.py --dry-run
```

Report: `scanners/artifacts/next_day_3_report.md`

### ShortHold80 detail

| | Backtest | Forward |
|---|---|---|
| Target | **+10% MFE** | same |
| Hold | **30 trading days** (~1.5 months) | same |
| Tip WR | **~96.7%** | **~88.7%** (Wilson~85%) |
| Signals | ~1.1 tips/day avg | **~1.4 tips/day ≈ 24–30 / month** |
| Rule | Calibrated GBM `P≥0.82` (max 3) | walk-forward |

Tips cluster (~17% of days fire). Quiet days are normal.

Each ShortHold run prints/emails the **last 10 signals** with Bias, Entry, Exit (target), and Reason.

```bash
# production (emails)
SENDER_EMAIL=... SENDER_PASSWORD=... RECEIVER_EMAIL=... python run_short_hold_80_scanner.py
```

Policy: `scanners/artifacts/short_hold_80_policy.json`  
Model: `scanners/artifacts/scanner_short_hold_80_model.joblib`

## On-time setup (free)

Use [cron-job.org](https://cron-job.org) → `workflow_dispatch` (GitHub cron is often late):

| Job | UTC cron | IST | Body |
|---|---|---|---|
| Morning | `37 3 * * 1-5` | 09:07 | `{"ref":"main","inputs":{"phase":"morning"}}` |
| Positional (ShortHold + NextDay3) | `15 10 * * 1-5` | 15:45 | `{"ref":"main","inputs":{"phase":"positional"}}` |
| Evening | `0 11 * * 1-5` | 16:30 | `{"ref":"main","inputs":{"phase":"evening"}}` |

POST `https://api.github.com/repos/chaitanyajerripothula/trade_bot/actions/workflows/main.yml/dispatches`  
Headers: `Authorization: Bearer <PAT>`, `Accept: application/vnd.github+json`, `Content-Type: application/json`

Repo secrets: `SENDER_EMAIL`, `SENDER_PASSWORD`, `RECEIVER_EMAIL`.

## Manual local run

```bash
pip install -r requirements.txt
python generate_adv.py
SENDER_EMAIL=... SENDER_PASSWORD=... RECEIVER_EMAIL=... python run_morning_scanners.py
SENDER_EMAIL=... SENDER_PASSWORD=... RECEIVER_EMAIL=... python run_short_hold_80_scanner.py
SENDER_EMAIL=... SENDER_PASSWORD=... RECEIVER_EMAIL=... python run_short_hold_8_scanner.py
SENDER_EMAIL=... SENDER_PASSWORD=... RECEIVER_EMAIL=... python run_short_hold_12_scanner.py
SENDER_EMAIL=... SENDER_PASSWORD=... RECEIVER_EMAIL=... python run_short_hold_15_scanner.py
```

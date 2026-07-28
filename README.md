# trade_bot

NSE F&O scanner suite:
- **Morning** — pre-open high-conviction bias (one mail)
- **Evening** — 20-day ADV harvest
- **Positional** — swing setups for **NSE monthly stock options** (separate mail with reason + entry/exit)

## Pre-open scanners (`run_morning_scanners.py`)

| File | Signal |
|---|---|
| `scanners/footprint_scanner.py` | Matched vol ≥ 5% ADV → Bias |
| `scanners/gap_scanner.py` | \|Gap\| ≥ 2% → Bias |
| `scanners/imbalance_scanner.py` | \|Imbalance\| ≥ 55% (book ≥ 20k) → Bias |
| `scanners/volume_shock_scanner.py` | ≥2% ADV + ≥50k matched → Bias |
| `scanners/near_52w_scanner.py` | Within 3% of 52W H/L → Bias |

## Positional scanners (`run_positional_scanners.py`)

Built for **stock options that expire on the last Thursday of the month**. Fresh swings require **DTE ≥ 8**. High conviction = **≥2 scanners** agree on CALL or PUT. Separate email includes why chosen, entry, SL, T1/T2.

| File | Signal |
|---|---|
| `scanners/positional_trend_scanner.py` | EMA stack + RSI swing zone + volume |
| `scanners/positional_breakout_scanner.py` | 20D high/low break + volume surge |
| `scanners/positional_pullback_scanner.py` | Trend intact + RSI reset toward EMA20 |
| `scanners/positional_momentum_scanner.py` | Strong directional day + orderly ATR |

Trade plan on underlying: **Entry = close**, **SL = 1.5×ATR**, **T1 = 2.5×ATR**, **T2 = 4×ATR**, plus suggested monthly ATM CE/PE.

```bash
python run_positional_scanners.py
python -m scanners.positional_trend_scanner
```

## On-time setup (free)

Use [cron-job.org](https://cron-job.org) → `workflow_dispatch` (GitHub cron is often late):

| Job | UTC cron | IST | Body |
|---|---|---|---|
| Morning | `37 3 * * 1-5` | 09:07 | `{"ref":"main","inputs":{"phase":"morning"}}` |
| Positional | `15 10 * * 1-5` | 15:45 | `{"ref":"main","inputs":{"phase":"positional"}}` |
| Evening | `0 11 * * 1-5` | 16:30 | `{"ref":"main","inputs":{"phase":"evening"}}` |

POST `https://api.github.com/repos/chaitanyajerripothula/trade_bot/actions/workflows/main.yml/dispatches`  
Headers: `Authorization: Bearer <PAT>`, `Accept: application/vnd.github+json`, `Content-Type: application/json`

Repo secrets: `SENDER_EMAIL`, `SENDER_PASSWORD`, `RECEIVER_EMAIL`.

## Manual local run

```bash
pip install -r requirements.txt
python generate_adv.py
SENDER_EMAIL=... SENDER_PASSWORD=... RECEIVER_EMAIL=... python run_morning_scanners.py
SENDER_EMAIL=... SENDER_PASSWORD=... RECEIVER_EMAIL=... python run_positional_scanners.py
```

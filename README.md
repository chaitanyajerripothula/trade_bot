# trade_bot

NSE F&O scanner suite:
- **Morning** — pre-open high-conviction bias (one mail)
- **Evening** — 20-day ADV harvest
- **Positional** — **1:20 R:R** lottery tickets for **NSE monthly stock options** (separate mail with reason + entry/exit)

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

## Positional scanners (`run_positional_scanners.py`)

Built for **stock options that expire on the last Thursday of the month**. Fresh swings require **DTE ≥ 15** (else rolls to next month). **1:20 only**: Trend + Momentum + Breakout must agree, ATR ≥ 2.5% of price, max **3** names. SL=1×ATR · T1=5×ATR (scale) · T2=20×ATR. Suggests OTM (~0.5×ATR) monthly options. Most tickets fail — size as lottery.

| File | Signal |
|---|---|
| `scanners/positional_trend_scanner.py` | EMA stack + RSI zone + volume |
| `scanners/positional_breakout_scanner.py` | Close vs **prior** 20D high/low + volume surge |
| `scanners/positional_pullback_scanner.py` | Trend intact + RSI reset toward EMA20 (optional 4th) |
| `scanners/positional_momentum_scanner.py` | Strong directional day + orderly ATR |

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

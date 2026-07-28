# trade_bot

NSE F&O scanner suite:
- **Morning** — pre-open high-conviction bias (one mail)
- **Evening** — 20-day ADV harvest
<<<<<<< HEAD
- **Positional** — **ShortHold80** only (+10% / ≤30 trading days, ≥80% tip WR)
=======
- **Positional / home runs** — **ZERO/0DTE** or **MONTHLY** options lotteries with very high conviction (separate mail)
>>>>>>> origin/main

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

<<<<<<< HEAD
## ShortHold80 (`run_short_hold_80_scanner.py`)

Only positional scanner. Max hold **1–2 months**:

| | Backtest | Forward |
|---|---|---|
| Target | **+10% MFE** | same |
| Hold | **30 trading days** (~1.5 months) | same |
| Tip WR | **~96.7%** | **~88.7%** (Wilson~85%) |
| Signals | ~1.1 tips/day avg | **~1.4 tips/day ≈ 24–30 / month** |
| Rule | Calibrated GBM `P≥0.82` (max 3) | walk-forward |

Tips cluster (~17% of days fire). Quiet days are normal.

```bash
# production (emails)
SENDER_EMAIL=... SENDER_PASSWORD=... RECEIVER_EMAIL=... python run_short_hold_80_scanner.py

# local check
python run_short_hold_80_scanner.py --dry-run
```

Policy: `scanners/artifacts/short_hold_80_policy.json`  
Report: `scanners/artifacts/short_hold_80_report.md`  
Model: `scanners/artifacts/scanner_short_hold_80_model.joblib`
=======
## Home-run scanners (`run_homerun_scanners.py`)

Pivot: **ZERO/0DTE (or near-zero DTE)** **or** **MONTHLY** expiry — home-run tickets only, very high conviction, tiny size.

| Bucket | What | Gate |
|---|---|---|
| **ZERO / 0DTE** | Index (NIFTY / BANKNIFTY / FINNIFTY) same-day / nearest weekly lottery | Day thrust ≥1.25%, vol ≥2×, break prior 20D extreme, RSI zone — max 2 |
| **ZERO monthly tail** | Stock options with **DTE ≤ 2** before this month’s last Thursday | ≥2.5% day + vol ≥2.5× + prior-range break + ATR≥3% — max 1 |
| **MONTHLY** | Stock options, last Thursday, **DTE ≥ 15** | Trend + Momentum + Breakout agree, ATR≥2.5%, 1×ATR SL / 20×ATR T2 — max 2 |

```bash
python run_homerun_scanners.py
# alias:
python run_positional_scanners.py
```

Research note: predicting ±15–20% swings at ≥80% tip win-rate on technicals alone was not attainable (see `scanners/artifacts/rl_20pct_report.md`). This suite does **not** claim that — it only fires sparse lottery tickets.

## Hero-or-zero ≥80% WR (`run_hero_or_zero_scanner.py`)

Deep Yahoo study (`research/hero_or_zero_deep.py`) swept ATR path geometries + gates:

- **≥80% train & test WR:** found (resolved-only: scratch if neither stop nor target hits)
- **≥80% with RR≥1.5:** **none** — large-RR “heroes” do not clear 80%
- Rules that work: **target 0.75–1.5R vs stop 1.5–2R** (RR≈0.5–0.75), hold 1–3d, gates like `thrust_bull` / `break_bull`

Report: `scanners/artifacts/hero_or_zero_report.md`.

```bash
SENDER_EMAIL=... SENDER_PASSWORD=... RECEIVER_EMAIL=... python run_hero_or_zero_scanner.py
```

## ShortHold80 scanner (`run_short_hold_80_scanner.py`)

Primary positional short-hold scanner under a **1–2 month** max hold:

| | Backtest | Forward |
|---|---|---|
| Target | **+10% MFE** | same |
| Hold | **30 trading days** (~1.5 months) | same |
| Tip WR | **~96.7%** | **~88.7%** (Wilson~85%) |
| Signals | ~1.1 tips/day avg | **~1.4 tips/day ≈ 24–30 / month** |
| Rule | Calibrated GBM `P≥0.82` (max 3) | walk-forward |

Tips cluster (~17% of days fire). Quiet days are normal.  
**+20% @ 80% WR is not attainable** inside this hold cap with this frequency.

```bash
# production (emails)
SENDER_EMAIL=... SENDER_PASSWORD=... RECEIVER_EMAIL=... python run_short_hold_80_scanner.py

# local check
python run_short_hold_80_scanner.py --dry-run
```

Policy: `scanners/artifacts/short_hold_80_policy.json` · Report: `scanners/artifacts/scanner20pct_80wr_report.md`.  
Alias: `run_twenty_pct_80wr_scanner.py` → same scanner.

## TenPct80 scanner (`run_tenpct_80wr_scanner.py`)

Older sparse research scanner for **+10% / ≤60d** (not the primary live short-hold product). Prefer **ShortHold80** above.

## Study: +20% in 15 days @ 80% tip WR?

Focused walk-forward research (`scanners/artifacts/move20_in_15d_report.md`):

- Base rate ≈ **4.7%** (test ≈ **2.6%**)
- Best calibrated tip precision ≈ **21%**
- Best technical gate ≈ **16%**
- **≥80% tip WR: not attainable** on Yahoo F&O technicals for a **15-day** hold
>>>>>>> origin/main

## On-time setup (free)

Use [cron-job.org](https://cron-job.org) → `workflow_dispatch` (GitHub cron is often late):

| Job | UTC cron | IST | Body |
|---|---|---|---|
| Morning | `37 3 * * 1-5` | 09:07 | `{"ref":"main","inputs":{"phase":"morning"}}` |
<<<<<<< HEAD
| Positional (ShortHold80) | `15 10 * * 1-5` | 15:45 | `{"ref":"main","inputs":{"phase":"positional"}}` |
=======
| Positional / home run | `15 10 * * 1-5` | 15:45 | `{"ref":"main","inputs":{"phase":"positional"}}` |
>>>>>>> origin/main
| Evening | `0 11 * * 1-5` | 16:30 | `{"ref":"main","inputs":{"phase":"evening"}}` |

POST `https://api.github.com/repos/chaitanyajerripothula/trade_bot/actions/workflows/main.yml/dispatches`  
Headers: `Authorization: Bearer <PAT>`, `Accept: application/vnd.github+json`, `Content-Type: application/json`

Repo secrets: `SENDER_EMAIL`, `SENDER_PASSWORD`, `RECEIVER_EMAIL`.

## Manual local run

```bash
pip install -r requirements.txt
python generate_adv.py
SENDER_EMAIL=... SENDER_PASSWORD=... RECEIVER_EMAIL=... python run_morning_scanners.py
<<<<<<< HEAD
SENDER_EMAIL=... SENDER_PASSWORD=... RECEIVER_EMAIL=... python run_short_hold_80_scanner.py
=======
SENDER_EMAIL=... SENDER_PASSWORD=... RECEIVER_EMAIL=... python run_homerun_scanners.py
>>>>>>> origin/main
```

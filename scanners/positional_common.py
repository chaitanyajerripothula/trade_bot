"""Shared helpers for positional / swing scanners (NSE monthly stock options)."""

from __future__ import annotations

import os
import smtplib
from datetime import date, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import StringIO

import numpy as np
import pandas as pd
import yfinance as yf

from fo_adv import DEFAULT_HEADERS, FO_LOT_CSV_URLS, clean_symbols, make_session

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Stock options on NSE expire on the last Thursday of the month.
# Red-team 1:20 design: tiny defined risk, asymmetric moonshot target.
MIN_DTE_FOR_NEW_SWING = 15  # 20R moves need calendar time
ATR_SL_MULT = 1.0           # risk unit = 1×ATR
ATR_T1_MULT = 5.0           # scale-out at 1:5 (optional)
ATR_T2_MULT = 20.0          # true home-run target = 1:20 vs SL
TARGET_RR = 20.0
MIN_ALIGNED_SCANNERS = 3    # Trend + Momentum + Breakout (Pullback optional 4th)
REQUIRE_TREND_ANCHOR = True
REQUIRE_POWER_SCANNER = True
# Both power scanners must agree — Trend+Pullback+one-power is not enough for 1:20.
REQUIRE_BOTH_POWER = True
MIN_ATR_PCT = 2.5           # need real volatility for large R multiples
MIN_VOL_RATIO = 1.5
MAX_HOMERUNS = 3            # very few lottery tickets
POWER_SCANNERS = frozenset({"Momentum", "Breakout"})
REQUIRED_SCANNERS = frozenset({"Trend", "Momentum", "Breakout"})

COMBINED_COLUMNS = [
    "Symbol",
    "Bias",
    "Option",
    "Conviction",
    "Scanners",
    "Reason",
    "Entry",
    "StopLoss",
    "Target1",
    "Target2",
    "RiskReward",
    "ATR",
    "RSI",
    "DTE",
    "Expiry",
]


def require_email_secrets() -> tuple[str, str, str]:
    sender = os.environ.get("SENDER_EMAIL")
    password = os.environ.get("SENDER_PASSWORD")
    receiver = os.environ.get("RECEIVER_EMAIL")
    if not all([sender, password, receiver]):
        raise SystemExit(
            "Missing SENDER_EMAIL, SENDER_PASSWORD, or RECEIVER_EMAIL environment variables."
        )
    return sender, password, receiver


def next_monthly_expiry(today: date | None = None) -> date:
    """NSE stock options: last Thursday of the month used for new swings."""
    today = today or date.today()

    def last_thursday(year: int, month: int) -> date:
        if month == 12:
            nxt = date(year + 1, 1, 1)
        else:
            nxt = date(year, month + 1, 1)
        d = nxt - timedelta(days=1)
        while d.weekday() != 3:  # Thursday
            d -= timedelta(days=1)
        return d

    exp = last_thursday(today.year, today.month)
    # If current monthly expiry is too close, roll to next month's last Thursday.
    if (exp - today).days < MIN_DTE_FOR_NEW_SWING:
        if exp.month == 12:
            exp = last_thursday(exp.year + 1, 1)
        else:
            exp = last_thursday(exp.year, exp.month + 1)
    return exp


def days_to_expiry(today: date | None = None) -> tuple[date, int]:
    today = today or date.today()
    exp = next_monthly_expiry(today)
    return exp, (exp - today).days


def suggest_atm_strike(price: float) -> float:
    """Rough ATM strike rounding used on NSE stock options."""
    if price >= 2000:
        step = 50
    elif price >= 500:
        step = 10
    elif price >= 100:
        step = 5
    else:
        step = 2.5
    return round(price / step) * step


def suggest_homerun_strike(
    price: float, bias: str, atr: float | None = None
) -> tuple[float, float, str]:
    """
    For 1:20 asymmetric payoff, prefer ~0.5×ATR OTM (rounded to strike step).
    Fixed 2-step OTM is too shallow on high-priced names and too deep on cheap ones.
    """
    atm = suggest_atm_strike(price)
    if price >= 2000:
        step = 50
    elif price >= 500:
        step = 10
    elif price >= 100:
        step = 5
    else:
        step = 2.5
    # At least 2 steps OTM; scale with ATR so lottery premium needs a real move.
    atr_steps = max(2, int(round((0.5 * float(atr)) / step))) if atr and atr > 0 else 2
    if bias == "CALL":
        otm = atm + atr_steps * step
        return (
            atm,
            otm,
            f"CALL OTM ~{otm:.0f} (ATM {atm:.0f}, ~{atr_steps} steps) — lottery for ~1:20 premium",
        )
    otm = max(step, atm - atr_steps * step)
    return (
        atm,
        otm,
        f"PUT OTM ~{otm:.0f} (ATM {atm:.0f}, ~{atr_steps} steps) — lottery for ~1:20 premium",
    )


def load_fo_symbols() -> list[str]:
    if os.path.exists("fno_adv.csv"):
        df = pd.read_csv("fno_adv.csv")
        return clean_symbols(df["Symbol"].tolist())
    session = make_session()
    last_err = None
    for url in FO_LOT_CSV_URLS:
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            if resp.content[:4] == b"%PDF":
                continue
            lot_df = pd.read_csv(StringIO(resp.text))
            lot_df.columns = [str(c).strip() for c in lot_df.columns]
            col = next(c for c in lot_df.columns if c.upper() == "SYMBOL")
            return clean_symbols(lot_df[col].tolist())
        except Exception as err:
            last_err = err
    raise SystemExit(f"Could not load F&O universe for positional scan: {last_err}")


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def build_positional_frame(period: str = "1y") -> pd.DataFrame:
    """
    Batch Yahoo daily bars for F&O equities and compute swing indicators.
    One row per symbol (latest complete session).

    Use ≥1y so EMA200 is a real 200-span average (6mo ≈126 bars silently
    fell back to EMA50 and weakened the Trend filter).
    """
    symbols = load_fo_symbols()
    tickers = [f"{s}.NS" for s in symbols]
    print(f"📦 Positional universe: {len(tickers)} F&O names — downloading {period} bars...")
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=False,
    )
    expiry, dte = days_to_expiry()
    rows = []

    for sym, ticker in zip(symbols, tickers):
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    continue
                bar = raw[ticker].dropna(how="all").copy()
            else:
                bar = raw.dropna(how="all").copy()
            if len(bar) < 60 or "Close" not in bar.columns:
                continue

            close = bar["Close"]
            high = bar["High"]
            low = bar["Low"]
            vol = bar["Volume"]
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            ema200 = close.ewm(span=200, adjust=False).mean() if len(bar) >= 200 else ema50
            rsi = _rsi(close, 14)
            atr = _atr(high, low, close, 14)
            vol_ma20 = vol.rolling(20).mean()
            high20 = high.rolling(20).max()
            low20 = low.rolling(20).min()
            # Prior 20D extremes exclude today — avoids breakout look-ahead bias.
            prev_high20 = high.shift(1).rolling(20).max()
            prev_low20 = low.shift(1).rolling(20).min()

            i = -1
            px = float(close.iloc[i])
            atr_v = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else np.nan
            rsi_v = float(rsi.iloc[i]) if pd.notna(rsi.iloc[i]) else np.nan
            if np.isnan(atr_v) or atr_v <= 0 or np.isnan(rsi_v):
                continue
            ph20 = float(prev_high20.iloc[i]) if pd.notna(prev_high20.iloc[i]) else float(high20.iloc[i])
            pl20 = float(prev_low20.iloc[i]) if pd.notna(prev_low20.iloc[i]) else float(low20.iloc[i])

            rows.append(
                {
                    "Symbol": sym,
                    "Close": px,
                    "Open": float(bar["Open"].iloc[i]),
                    "High": float(high.iloc[i]),
                    "Low": float(low.iloc[i]),
                    "Volume": float(vol.iloc[i]),
                    "VolMA20": float(vol_ma20.iloc[i]) if pd.notna(vol_ma20.iloc[i]) else np.nan,
                    "EMA20": float(ema20.iloc[i]),
                    "EMA50": float(ema50.iloc[i]),
                    "EMA200": float(ema200.iloc[i]),
                    "RSI": rsi_v,
                    "ATR": atr_v,
                    "High20": float(high20.iloc[i]),
                    "Low20": float(low20.iloc[i]),
                    "PrevHigh20": ph20,
                    "PrevLow20": pl20,
                    "PrevClose": float(close.iloc[-2]),
                    "RSI_5d_ago": float(rsi.iloc[-6]) if len(rsi) >= 6 and pd.notna(rsi.iloc[-6]) else np.nan,
                    "EMA20_5d_ago": float(ema20.iloc[-6]) if len(ema20) >= 6 else np.nan,
                    "DTE": dte,
                    "Expiry": expiry.isoformat(),
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Positional frame empty — Yahoo data unavailable.")
    df["VolRatio"] = df["Volume"] / df["VolMA20"].replace(0, np.nan)
    df["ExtATR"] = (df["Close"] - df["EMA20"]).abs() / df["ATR"]
    print(f"✅ Positional frame ready: {len(df)} symbols | monthly expiry {expiry} | DTE={dte}")
    return df


def build_trade_plan(row: pd.Series, bias: str, reason: str) -> dict | None:
    """
    Asymmetric 1:20 plan on the underlying.

    Risk = 1×ATR. Home-run target = 20×ATR (R:R 1:20). T1 at 5×ATR is an
    optional scale-out so the whole ticket is not binary moonshot-or-zero.

    Returns None when geometry is impossible (e.g. PUT T2 would be ≤0).
    """
    entry = float(row["Close"])
    atr = float(row["ATR"])
    if atr <= 0 or entry <= 0:
        return None
    if bias == "CALL":
        stop = entry - ATR_SL_MULT * atr
        t1 = entry + ATR_T1_MULT * atr
        t2 = entry + ATR_T2_MULT * atr
        if stop <= 0:
            return None
    else:
        stop = entry + ATR_SL_MULT * atr
        t1 = entry - ATR_T1_MULT * atr
        t2 = entry - ATR_T2_MULT * atr
        # 20×ATR below spot can go through zero on cheap/high-ATR names — skip.
        if t2 <= 0 or t1 <= 0:
            return None
    risk = abs(entry - stop)
    reward_hr = abs(t2 - entry)
    rr = round(reward_hr / risk, 2) if risk > 0 else 0
    if rr < TARGET_RR * 0.99:
        return None
    atm, otm, option = suggest_homerun_strike(entry, bias, atr=atr)
    return {
        "Symbol": row["Symbol"],
        "Bias": bias,
        "Option": f"{option} | monthly {row['Expiry']}",
        "Reason": reason,
        "Entry": round(entry, 2),
        "StopLoss": round(stop, 2),
        "Target1": round(t1, 2),
        "Target2": round(t2, 2),
        "RiskReward": rr,
        "ATR": round(atr, 2),
        "RSI": round(float(row["RSI"]), 1),
        "DTE": int(row["DTE"]),
        "Expiry": row["Expiry"],
        "ATM_Strike": atm,
        "OTM_Strike": otm,
    }


def expiry_ok(df: pd.DataFrame) -> pd.DataFrame:
    """Skip new positional options swings too close to monthly stock-option expiry."""
    return df[df["DTE"] >= MIN_DTE_FOR_NEW_SWING].copy()


def club_positional(scanner_hits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    1:20 HOME RUN filter.

    Keep only symbols where:
      - Trend + Momentum + Breakout all agree on CALL/PUT,
      - Pullback may add a 4th confirm but cannot substitute for power,
      - ATR ≥ MIN_ATR_PCT of price,
      - plan still has ≥1:20 geometry,
      - then keep top MAX_HOMERUNS by score.
    """
    pieces = []
    for name, hits in scanner_hits.items():
        if hits is None or hits.empty:
            continue
        part = hits.copy()
        part["Scanner"] = name
        pieces.append(part)

    if not pieces:
        return pd.DataFrame(columns=COMBINED_COLUMNS)

    stacked = pd.concat(pieces, ignore_index=True)
    rows = []
    for symbol, grp in stacked.groupby("Symbol"):
        call_n = int((grp["Bias"] == "CALL").sum())
        put_n = int((grp["Bias"] == "PUT").sum())
        if call_n == 0 and put_n == 0:
            continue
        if call_n >= put_n and call_n >= MIN_ALIGNED_SCANNERS:
            bias, conviction = "CALL", call_n
            aligned = grp[grp["Bias"] == "CALL"]
        elif put_n >= MIN_ALIGNED_SCANNERS:
            bias, conviction = "PUT", put_n
            aligned = grp[grp["Bias"] == "PUT"]
        else:
            continue

        scanners = set(aligned["Scanner"].unique().tolist())
        if REQUIRE_TREND_ANCHOR and "Trend" not in scanners:
            continue
        if REQUIRE_BOTH_POWER and not POWER_SCANNERS.issubset(scanners):
            continue
        if REQUIRE_POWER_SCANNER and not (POWER_SCANNERS & scanners):
            continue
        if not REQUIRED_SCANNERS.issubset(scanners):
            continue

        best = aligned.sort_values(by="RiskReward", ascending=False).iloc[0]
        atr_pct = (float(best["ATR"]) / float(best["Entry"]) * 100) if best["Entry"] else 0
        if atr_pct < MIN_ATR_PCT:
            continue
        # Reject plans that somehow lost the 1:20 geometry.
        if float(best["RiskReward"]) < TARGET_RR * 0.99:
            continue

        reasons = [f"[{r['Scanner']}] {r['Reason']}" for _, r in aligned.iterrows()]
        score = conviction * 10 + atr_pct + float(best["RiskReward"])
        rows.append(
            {
                "Symbol": symbol,
                "Bias": bias,
                "Option": best["Option"],
                "Conviction": conviction,
                "Scanners": ", ".join(sorted(scanners)),
                "Reason": " | ".join(reasons),
                "Entry": best["Entry"],
                "StopLoss": best["StopLoss"],
                "Target1": best["Target1"],
                "Target2": best["Target2"],
                "RiskReward": best["RiskReward"],
                "ATR": best["ATR"],
                "RSI": best["RSI"],
                "DTE": best["DTE"],
                "Expiry": best["Expiry"],
                "ATR_Pct": round(atr_pct, 2),
                "Score": round(score, 2),
            }
        )

    if not rows:
        return pd.DataFrame(columns=COMBINED_COLUMNS)

    out = pd.DataFrame(rows).sort_values(by=["Score", "Conviction"], ascending=False)
    out = out.head(MAX_HOMERUNS).reset_index(drop=True)
    return out


def send_positional_email(combined: pd.DataFrame) -> None:
    if combined.empty:
        print("No 1:20 HOME RUN positional setups — suppressing email.")
        return

    sender, password, receiver = require_email_secrets()
    blocks = []
    for _, r in combined.iterrows():
        direction = "bullish 1:20 HOME RUN (buy OTM CE)" if r["Bias"] == "CALL" else "bearish 1:20 HOME RUN (buy OTM PE)"
        atr_pct = r["ATR_Pct"] if "ATR_Pct" in r and pd.notna(r["ATR_Pct"]) else ""
        blocks.append(
            f"""
            <div style="border:2px solid #111; padding:14px; margin:12px 0;">
              <h3 style="margin:0 0 8px 0;">1:20 HOME RUN {r['Symbol']} — {r['Bias']} / {direction}</h3>
              <p><b>Why chosen:</b> {r['Reason']}</p>
              <p><b>Confluence:</b> {r['Scanners']} (conviction {r['Conviction']}/4)
                 · ATR room {atr_pct}% · planned R:R {r['RiskReward']}:1 · Score {r.get('Score', '')}</p>
              <p><b>Monthly options:</b> {r['Option']} · DTE {r['DTE']} · Expiry {r['Expiry']}</p>
              <table>
                <tr>
                  <th>Entry</th><th>Stop (1×ATR)</th>
                  <th>Scale-out T1 (5×ATR = 1:5)</th>
                  <th>Home-run T2 (20×ATR = 1:20)</th>
                  <th>R:R to T2</th><th>ATR</th><th>RSI</th>
                </tr>
                <tr>
                  <td>{r['Entry']}</td><td>{r['StopLoss']}</td><td>{r['Target1']}</td>
                  <td>{r['Target2']}</td><td>{r['RiskReward']}</td><td>{r['ATR']}</td><td>{r['RSI']}</td>
                </tr>
              </table>
              <p style="color:#555;font-size:12px;">
                <b>Red-team honesty:</b> T2 is a low-probability moonshot (~20 ATR). Size tiny.
                Prefer the suggested OTM monthly option so premium payoff can approach 1:20 if T2 hits.
                Exit if underlying tags SL. Optional partial at T1; leave a runner for T2.
                Most tickets will stop out — that is expected for 1:20 geometry.
              </p>
            </div>
            """
        )

    subject = f"1:20 HOME RUN POSITIONAL — {len(combined)} monthly options tickets"
    html = f"""
    <html>
      <head>
        <style>
          body {{ font-family: sans-serif; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ padding: 8px; border-bottom: 1px solid #ddd; text-align: left; }}
          th {{ background: #1a1a1a; color: #fff; }}
        </style>
      </head>
      <body>
        <h2>1:20 HOME RUN Positional Scanners (NSE Monthly Stock Options)</h2>
        <p>
          Defined risk = <b>1×ATR</b>. Home-run target = <b>20×ATR (R:R 1:20)</b>.
          Optional scale-out at 5×ATR. Requires <b>Trend + Momentum + Breakout</b>
          (Pullback optional), ATR≥{MIN_ATR_PCT}% of price, DTE≥{MIN_DTE_FOR_NEW_SWING},
          max {MAX_HOMERUNS} names. Stock options expire last Thursday of the month.
        </p>
        {''.join(blocks)}
        <p><i>trade_bot positional 1:20 suite — separate from pre-open morning mail</i></p>
      </body>
    </html>
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
    print(f"📨 1:20 HOME RUN positional mail sent: {len(combined)} setups")

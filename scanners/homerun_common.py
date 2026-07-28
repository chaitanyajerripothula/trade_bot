"""
Home-run options tickets: 0DTE / near-zero DTE OR monthly expiry.

Two buckets (very high conviction, intentionally sparse):

1) MONTHLY  — NSE stock options, last Thursday, DTE ≥ 15
              Require Trend + Momentum + Breakout alignment (existing club).

2) ZERO     — 0DTE-style / near-zero DTE lottery
              a) Index proxies (NIFTY, BANKNIFTY, FINNIFTY) for same-day / nearest weekly
              b) Stock monthlies with DTE ≤ 2 (expiry week gamma lottery)

These are defined-risk lottery tickets, not 80%-WR swing forecasts.
"""

from __future__ import annotations

import smtplib
from datetime import date, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import yfinance as yf

from scanners.positional_common import (
    SMTP_PORT,
    SMTP_SERVER,
    build_positional_frame,
    build_trade_plan,
    club_positional,
    require_email_secrets,
)

# --- conviction knobs ---
MONTHLY_MAX = 2
ZERO_MAX = 2
NEAR_ZERO_STOCK_DTE = 2  # stock monthly in final 0–2 days
ZERO_DAY_CHG_PCT = 1.25
ZERO_VOL_RATIO = 2.0
ZERO_ATR_PCT = 0.8  # indices have lower ATR%
STOCK_NEAR_ZERO_CHG = 2.5
STOCK_NEAR_ZERO_VOL = 2.5
STOCK_NEAR_ZERO_ATR = 3.0

INDEX_UNIVERSE = (
    # symbol, yahoo ticker, strike step, label
    ("NIFTY", "^NSEI", 50, "Nifty 50"),
    ("BANKNIFTY", "^NSEBANK", 100, "Bank Nifty"),
    ("FINNIFTY", "NIFTY_FIN_SERVICE.NS", 50, "Fin Nifty"),
)


def last_thursday(year: int, month: int) -> date:
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    d = nxt - timedelta(days=1)
    while d.weekday() != 3:
        d -= timedelta(days=1)
    return d


def this_month_stock_expiry(today: date | None = None) -> tuple[date, int]:
    """Current calendar month's stock-option expiry (may be 0–few DTE)."""
    today = today or date.today()
    exp = last_thursday(today.year, today.month)
    return exp, (exp - today).days


def nearest_weekly_hint(today: date | None = None) -> str:
    """
    NSE index weeklies rotate; without an exchange calendar we label the ticket
    as 0DTE if today is a typical expiry weekday, else 'nearest weekly'.
    """
    today = today or date.today()
    # Common pattern (subject to exchange changes): Tue FinNifty, Wed Bank, Thu Nifty.
    if today.weekday() in (1, 2, 3):  # Tue/Wed/Thu
        return f"0DTE candidate ({today.isoformat()})"
    return f"nearest weekly (as of {today.isoformat()})"


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def scan_index_zero_dte() -> pd.DataFrame:
    """Ultra-strict same-day / 0DTE-style index lottery from Yahoo daily bars."""
    tickers = [t for _, t, _, _ in INDEX_UNIVERSE]
    raw = yf.download(
        tickers=tickers,
        period="6mo",
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=False,
    )
    expiry_hint = nearest_weekly_hint()
    rows = []
    for sym, ticker, step, label in INDEX_UNIVERSE:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    continue
                bar = raw[ticker].dropna(how="all").copy()
            else:
                bar = raw.dropna(how="all").copy()
            if len(bar) < 40:
                continue
            c = bar["Close"].astype(float)
            h = bar["High"].astype(float)
            l = bar["Low"].astype(float)
            v = bar["Volume"].astype(float)
            atr = _atr(h, l, c)
            rsi = _rsi(c)
            volma = v.rolling(20).mean()
            i = -1
            px = float(c.iloc[i])
            prev = float(c.iloc[i - 1])
            atr_v = float(atr.iloc[i])
            if atr_v <= 0 or px <= 0:
                continue
            chg = (px - prev) / prev * 100
            vol_ratio = float(v.iloc[i] / volma.iloc[i]) if volma.iloc[i] else np.nan
            atr_pct = atr_v / px * 100
            rsi_v = float(rsi.iloc[i])
            broke_hi = px > float(h.iloc[i - 20 : i].max())
            broke_lo = px < float(l.iloc[i - 20 : i].min())

            bias = None
            reason = None
            if (
                chg >= ZERO_DAY_CHG_PCT
                and vol_ratio >= ZERO_VOL_RATIO
                and atr_pct >= ZERO_ATR_PCT
                and broke_hi
                and 55 <= rsi_v <= 78
            ):
                bias = "CALL"
                reason = (
                    f"ZERO/0DTE index thrust +{chg:.1f}% on {vol_ratio:.1f}× vol; "
                    f"break prior 20D high; RSI {rsi_v:.0f}; {label}"
                )
            elif (
                chg <= -ZERO_DAY_CHG_PCT
                and vol_ratio >= ZERO_VOL_RATIO
                and atr_pct >= ZERO_ATR_PCT
                and broke_lo
                and 22 <= rsi_v <= 45
            ):
                bias = "PUT"
                reason = (
                    f"ZERO/0DTE index dump {chg:.1f}% on {vol_ratio:.1f}× vol; "
                    f"break prior 20D low; RSI {rsi_v:.0f}; {label}"
                )
            if not bias:
                continue

            atm = round(px / step) * step
            otm = atm + (2 * step if bias == "CALL" else -2 * step)
            # Same-session geometry: tight stop, home-run stretch target
            if bias == "CALL":
                stop = px - 0.75 * atr_v
                t1 = px + 2.0 * atr_v
                t2 = px + 4.0 * atr_v
            else:
                stop = px + 0.75 * atr_v
                t1 = px - 2.0 * atr_v
                t2 = px - 4.0 * atr_v
            risk = abs(px - stop)
            rr = round(abs(t2 - px) / risk, 2) if risk else 0
            rows.append(
                {
                    "Bucket": "ZERO_0DTE",
                    "Symbol": sym,
                    "Bias": bias,
                    "Option": f"{bias} OTM~{otm:.0f} (ATM {atm:.0f}) | {expiry_hint}",
                    "Conviction": 3,
                    "Scanners": "IndexThrust+Break+Vol",
                    "Reason": reason,
                    "Entry": round(px, 2),
                    "StopLoss": round(stop, 2),
                    "Target1": round(t1, 2),
                    "Target2": round(t2, 2),
                    "RiskReward": rr,
                    "ATR": round(atr_v, 2),
                    "ATR_Pct": round(atr_pct, 2),
                    "RSI": round(rsi_v, 1),
                    "DTE": 0 if "0DTE" in expiry_hint else 1,
                    "Expiry": expiry_hint,
                    "Score": round(abs(chg) * vol_ratio, 2),
                }
            )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(by="Score", ascending=False).head(ZERO_MAX)
    return out.reset_index(drop=True)


def scan_stock_near_zero(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Stock monthly options in the final 0–2 DTE before this month's last Thursday.
    Only extreme thrust — gamma lottery, max few names.
    """
    exp, dte = this_month_stock_expiry()
    if dte < 0 or dte > NEAR_ZERO_STOCK_DTE:
        print(f"[NEAR_ZERO] skip stocks — this-month expiry DTE={dte} (want 0–{NEAR_ZERO_STOCK_DTE})")
        return pd.DataFrame()

    df = frame if frame is not None else build_positional_frame(period="1y")
    # Rebuild DTE against THIS month expiry (positional frame may have rolled next month)
    df = df.copy()
    df["DTE"] = dte
    df["Expiry"] = exp.isoformat()
    atr_pct = df["ATR"] / df["Close"] * 100
    chg = (df["Close"] - df["PrevClose"]) / df["PrevClose"] * 100
    rows = []

    bull = df[
        (df["Close"] > df["PrevHigh20"])
        & (chg >= STOCK_NEAR_ZERO_CHG)
        & (df["VolRatio"] >= STOCK_NEAR_ZERO_VOL)
        & (atr_pct >= STOCK_NEAR_ZERO_ATR)
        & (df["RSI"] >= 58)
        & (df["RSI"] <= 80)
        & (df["EMA20"] > df["EMA50"])
    ]
    for _, r in bull.iterrows():
        plan = build_trade_plan(
            r,
            "CALL",
            f"NEAR-ZERO monthly CE lottery DTE={dte}: +{((r['Close']-r['PrevClose'])/r['PrevClose']*100):.1f}% "
            f"thru prior 20D high on {r['VolRatio']:.1f}× vol; expiry {exp}",
        )
        if not plan:
            continue
        plan["Bucket"] = "ZERO_MONTHLY_TAIL"
        plan["Conviction"] = 3
        plan["Scanners"] = "NearZeroThrust"
        plan["ATR_Pct"] = round(float(r["ATR"] / r["Close"] * 100), 2)
        plan["Score"] = round(
            abs(float(r["Close"] - r["PrevClose"]) / float(r["PrevClose"]) * 100) * float(r["VolRatio"]),
            2,
        )
        plan["DTE"] = dte
        plan["Expiry"] = exp.isoformat()
        rows.append(plan)

    bear = df[
        (df["Close"] < df["PrevLow20"])
        & (chg <= -STOCK_NEAR_ZERO_CHG)
        & (df["VolRatio"] >= STOCK_NEAR_ZERO_VOL)
        & (atr_pct >= STOCK_NEAR_ZERO_ATR)
        & (df["RSI"] <= 42)
        & (df["RSI"] >= 20)
        & (df["EMA20"] < df["EMA50"])
    ]
    for _, r in bear.iterrows():
        plan = build_trade_plan(
            r,
            "PUT",
            f"NEAR-ZERO monthly PE lottery DTE={dte}: {((r['Close']-r['PrevClose'])/r['PrevClose']*100):.1f}% "
            f"thru prior 20D low on {r['VolRatio']:.1f}× vol; expiry {exp}",
        )
        if not plan:
            continue
        plan["Bucket"] = "ZERO_MONTHLY_TAIL"
        plan["Conviction"] = 3
        plan["Scanners"] = "NearZeroThrust"
        plan["ATR_Pct"] = round(float(r["ATR"] / r["Close"] * 100), 2)
        plan["Score"] = round(
            abs(float(r["Close"] - r["PrevClose"]) / float(r["PrevClose"]) * 100) * float(r["VolRatio"]),
            2,
        )
        plan["DTE"] = dte
        plan["Expiry"] = exp.isoformat()
        rows.append(plan)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(by="Score", ascending=False).head(1)
    return out.reset_index(drop=True)


def monthly_homeruns(scanner_hits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Reuse positional club (Trend+Mom+Breakout) — cap tighter for home runs."""
    combined = club_positional(scanner_hits)
    if combined.empty:
        return combined
    combined = combined.head(MONTHLY_MAX).copy()
    combined["Bucket"] = "MONTHLY"
    return combined


def send_homerun_email(monthly: pd.DataFrame, zero: pd.DataFrame) -> None:
    if (monthly is None or monthly.empty) and (zero is None or zero.empty):
        print("No ZERO/MONTHLY home-run tickets — suppressing email.")
        return

    sender, password, receiver = require_email_secrets()

    def blocks(df: pd.DataFrame, title: str) -> list[str]:
        if df is None or df.empty:
            return [f"<p><b>{title}:</b> none today (gate held).</p>"]
        out = [f"<h3>{title}</h3>"]
        for _, r in df.iterrows():
            out.append(
                f"""
                <div style="border:2px solid #111;padding:12px;margin:10px 0;">
                  <h4 style="margin:0 0 6px 0;">{r.get('Symbol')} — {r.get('Bias')} · {r.get('Bucket', title)}</h4>
                  <p><b>Why:</b> {r.get('Reason')}</p>
                  <p><b>Confluence:</b> {r.get('Scanners')} · conviction {r.get('Conviction')}
                     · R:R {r.get('RiskReward')} · ATR% {r.get('ATR_Pct', '')}</p>
                  <p><b>Options:</b> {r.get('Option')} · DTE {r.get('DTE')} · {r.get('Expiry')}</p>
                  <table>
                    <tr><th>Entry</th><th>Stop</th><th>T1</th><th>T2 / stretch</th><th>R:R</th><th>RSI</th></tr>
                    <tr>
                      <td>{r.get('Entry')}</td><td>{r.get('StopLoss')}</td>
                      <td>{r.get('Target1')}</td><td>{r.get('Target2')}</td>
                      <td>{r.get('RiskReward')}</td><td>{r.get('RSI')}</td>
                    </tr>
                  </table>
                  <p style="color:#555;font-size:12px;">
                    Lottery sizing only. ZERO/0DTE can expire worthless same day.
                    MONTHLY uses OTM structure for asymmetric payoff. Not advice.
                  </p>
                </div>
                """
            )
        return out

    n = (0 if monthly is None or monthly.empty else len(monthly)) + (
        0 if zero is None or zero.empty else len(zero)
    )
    subject = f"HOME RUN — {n} ticket(s) [ZERO/0DTE + MONTHLY]"
    html = f"""
    <html>
      <head>
        <style>
          body {{ font-family: sans-serif; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ padding: 8px; border-bottom: 1px solid #ddd; text-align: left; }}
          th {{ background: #111; color: #fff; }}
        </style>
      </head>
      <body>
        <h2>Home runs — ZERO/0DTE or MONTHLY expiry</h2>
        <p>
          Very high conviction only. Two buckets:
          <b>ZERO</b> (0DTE / near-zero DTE gamma lottery) and
          <b>MONTHLY</b> (stock options, last Thursday, DTE≥15, Trend+Momentum+Breakout).
          Expect most tickets to fail — size tiny.
        </p>
        {''.join(blocks(zero, 'ZERO / 0DTE / near-zero'))}
        {''.join(blocks(monthly, 'MONTHLY stock options'))}
        <p><i>trade_bot home-run suite — pivoted off RL 15% swing predictor</i></p>
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
    print(f"📨 Home-run mail sent: {n} ticket(s)")

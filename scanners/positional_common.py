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
MIN_DTE_FOR_NEW_SWING = 8
ATR_SL_MULT = 1.5
ATR_T1_MULT = 3.0   # home-run T1
ATR_T2_MULT = 5.0   # home-run T2
MIN_ALIGNED_SCANNERS = 2  # Trend + power confirmation
REQUIRE_TREND_ANCHOR = True  # structure must agree
REQUIRE_POWER_SCANNER = True  # Momentum and/or Breakout must agree (not Pullback alone)
MIN_ATR_PCT = 2.0  # ATR must be ≥2% of price (options need room)
MIN_VOL_RATIO = 1.3
MAX_HOMERUNS = 5
POWER_SCANNERS = frozenset({"Momentum", "Breakout"})

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


def build_positional_frame(period: str = "6mo") -> pd.DataFrame:
    """
    Batch Yahoo daily bars for F&O equities and compute swing indicators.
    One row per symbol (latest complete session).
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

            i = -1
            px = float(close.iloc[i])
            atr_v = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else np.nan
            rsi_v = float(rsi.iloc[i]) if pd.notna(rsi.iloc[i]) else np.nan
            if np.isnan(atr_v) or atr_v <= 0 or np.isnan(rsi_v):
                continue

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


def build_trade_plan(row: pd.Series, bias: str, reason: str) -> dict:
    """ATR-based entry/SL/targets for monthly options swing on the underlying."""
    entry = float(row["Close"])
    atr = float(row["ATR"])
    if bias == "CALL":
        stop = entry - ATR_SL_MULT * atr
        t1 = entry + ATR_T1_MULT * atr
        t2 = entry + ATR_T2_MULT * atr
    else:
        stop = entry + ATR_SL_MULT * atr
        t1 = entry - ATR_T1_MULT * atr
        t2 = entry - ATR_T2_MULT * atr
    risk = abs(entry - stop)
    reward = abs(t1 - entry)
    rr = round(reward / risk, 2) if risk > 0 else 0
    atm = suggest_atm_strike(entry)
    option = f"{bias} ATM ~{atm:.0f} (monthly {row['Expiry']})"
    return {
        "Symbol": row["Symbol"],
        "Bias": bias,
        "Option": option,
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
    }


def expiry_ok(df: pd.DataFrame) -> pd.DataFrame:
    """Skip new positional options swings too close to monthly stock-option expiry."""
    return df[df["DTE"] >= MIN_DTE_FOR_NEW_SWING].copy()


def club_positional(scanner_hits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Home-run filter for monthly options swings.

    Keep only symbols where:
      - ≥3 scanners agree on CALL or PUT,
      - Trend scanner is one of them (structure anchor),
      - ATR ≥ MIN_ATR_PCT of price,
      - then rank and keep top MAX_HOMERUNS.
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

        scanners = sorted(aligned["Scanner"].unique().tolist())
        if REQUIRE_TREND_ANCHOR and "Trend" not in scanners:
            continue
        if REQUIRE_POWER_SCANNER and not (POWER_SCANNERS & set(scanners)):
            # Pullback+Trend alone is not a home run — need Momentum or Breakout thrust.
            continue

        best = aligned.sort_values(by="RiskReward", ascending=False).iloc[0]
        atr_pct = (float(best["ATR"]) / float(best["Entry"]) * 100) if best["Entry"] else 0
        if atr_pct < MIN_ATR_PCT:
            continue

        reasons = [f"[{r['Scanner']}] {r['Reason']}" for _, r in aligned.iterrows()]
        # Rank: confluence first, then volatility room, then R:R
        score = conviction * 10 + atr_pct + float(best["RiskReward"])
        rows.append(
            {
                "Symbol": symbol,
                "Bias": bias,
                "Option": best["Option"],
                "Conviction": conviction,
                "Scanners": ", ".join(scanners),
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
        print("No HOME RUN positional setups — suppressing email.")
        return

    sender, password, receiver = require_email_secrets()
    blocks = []
    for _, r in combined.iterrows():
        direction = "bullish HOME RUN (buy CE)" if r["Bias"] == "CALL" else "bearish HOME RUN (buy PE)"
        atr_pct = r["ATR_Pct"] if "ATR_Pct" in r and pd.notna(r["ATR_Pct"]) else ""
        blocks.append(
            f"""
            <div style="border:2px solid #111; padding:14px; margin:12px 0;">
              <h3 style="margin:0 0 8px 0;">HOME RUN {r['Symbol']} — {r['Bias']} / {direction}</h3>
              <p><b>Why this is a home run:</b> {r['Reason']}</p>
              <p><b>Confluence:</b> {r['Scanners']} (conviction {r['Conviction']}/4)
                 · ATR room {atr_pct}% of price · Score {r.get('Score', '')}</p>
              <p><b>Monthly options:</b> {r['Option']} · DTE {r['DTE']} · Expiry {r['Expiry']}</p>
              <table>
                <tr><th>Entry</th><th>Stop Loss</th><th>Target 1 (3×ATR)</th><th>Target 2 (5×ATR)</th><th>R:R</th><th>ATR</th><th>RSI</th></tr>
                <tr>
                  <td>{r['Entry']}</td><td>{r['StopLoss']}</td><td>{r['Target1']}</td>
                  <td>{r['Target2']}</td><td>{r['RiskReward']}</td><td>{r['ATR']}</td><td>{r['RSI']}</td>
                </tr>
              </table>
              <p style="color:#555;font-size:12px;">
                Only top {MAX_HOMERUNS} names with Trend + (Momentum or Breakout),
                ATR≥{MIN_ATR_PCT}% of price. Exit option at underlying SL; partial at T1; trail to T2.
              </p>
            </div>
            """
        )

    subject = f"HOME RUN POSITIONAL — {len(combined)} monthly options swings"
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
        <h2>HOME RUN Positional Scanners (NSE Monthly Stock Options)</h2>
        <p>
          Strict HOME RUN filter: Trend + (Momentum or Breakout), ATR ≥ {MIN_ATR_PCT}% of price,
          max {MAX_HOMERUNS} names. SL=1.5×ATR · T1=3×ATR · T2=5×ATR.
          Stock options expire last Thursday of the month.
        </p>
        {''.join(blocks)}
        <p><i>trade_bot positional HOME RUN suite — separate from pre-open morning mail</i></p>
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
    print(f"📨 HOME RUN positional mail sent: {len(combined)} setups")

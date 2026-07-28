"""
TwentyPct80 — daily positional tip: ≥+20% within hold, ≥80% tip WR, ≥1 stock/day.

Mode: **daily top-1** — every session, tip the single highest calibrated P(+20% MFE).

Validated (Yahoo F&O walk-forward, hold=180):
  Backtest: tip WR ≈ 99.3% (1.0 tips/day)
  Forward:  tip WR ≈ 88.7% (n=115, Wilson ≈ 82%, 1.0 tips/day)
  Latest 20% labeled: ≈ 91.9% (1.0 tips/day)

Note: a hard probability threshold at hold=60 that fires ≥1/day only reaches ~68% forward WR.
Daily top-1 at hold=180 keeps both the frequency constraint and ≥80% WR.
"""

from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from scanners.positional_common import (
    SMTP_PORT,
    SMTP_SERVER,
    days_to_expiry,
    load_fo_symbols,
    require_email_secrets,
    suggest_atm_strike,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "scanners" / "artifacts" / "scanner20pct_80wr_model.joblib"
SCANNER_NAME = "TwentyPct80"


def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing model {MODEL_PATH}. Run research/scanner20pct_80wr_nonlinear.py")
    return joblib.load(MODEL_PATH)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))


def _atr(high, low, close, period=14):
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _nifty_regime() -> pd.DataFrame:
    n = yf.download("^NSEI", period="2y", interval="1d", progress=False, auto_adjust=False)
    if isinstance(n.columns, pd.MultiIndex):
        if "Close" in n.columns.get_level_values(0):
            c = n["Close"]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[:, 0]
        else:
            c = n.xs("Close", axis=1, level=-1).iloc[:, 0]
    else:
        c = n["Close"]
    c = c.astype(float)
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(c.index).tz_localize(None),
            "nifty_bull": ((c > ema50) & (ema50 > ema200)).astype(int).values,
            "nifty_mom20": (c.pct_change(20) * 100).values,
            "nifty_mom60": (c.pct_change(60) * 100).values,
        }
    ).dropna()


def build_live_feature_frame(period: str = "2y") -> pd.DataFrame:
    symbols = load_fo_symbols()
    tickers = [f"{s}.NS" for s in symbols]
    print(f"📦 TwentyPct80 universe: {len(tickers)} — {period}")
    raw = yf.download(
        tickers=tickers, period=period, interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=False
    )
    reg = _nifty_regime()
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
            if len(bar) < 260:
                continue
            c = bar["Close"].astype(float)
            h = bar["High"].astype(float)
            l = bar["Low"].astype(float)
            o = bar["Open"].astype(float)
            v = bar["Volume"].astype(float)
            ema20 = c.ewm(span=20, adjust=False).mean()
            ema50 = c.ewm(span=50, adjust=False).mean()
            ema200 = c.ewm(span=200, adjust=False).mean()
            atr = _atr(h, l, c)
            rsi = _rsi(c)
            volma = v.rolling(20).mean()
            prev_hi = h.shift(1).rolling(20).max()
            prev_lo = l.shift(1).rolling(20).min()
            hi52 = h.rolling(252).max()
            lo52 = l.rolling(252).min()
            rng20 = (h.rolling(20).max() - l.rolling(20).min()) / c.replace(0, np.nan)
            rng5 = (h.rolling(5).max() - l.rolling(5).min()) / c.replace(0, np.nan)
            bull = (c > ema20) & (ema20 > ema50) & (c > ema200)
            bear = (c < ema20) & (ema20 < ema50) & (c < ema200)
            trend = np.where(bull, 2, np.where(bear, -2, np.where(c > ema50, 1, np.where(c < ema50, -1, 0))))
            brk = np.where(c > prev_hi, 1, np.where(c < prev_lo, -1, 0))
            idx = pd.to_datetime(bar.index).tz_localize(None)
            df = pd.DataFrame(
                {
                    "Date": idx,
                    "Close": c.to_numpy(),
                    "Open": o.to_numpy(),
                    "High": h.to_numpy(),
                    "Low": l.to_numpy(),
                    "ATR": atr.to_numpy(),
                    "trend": trend,
                    "rsi": rsi.to_numpy(),
                    "atr_pct": (atr / c * 100).to_numpy(),
                    "vol_ratio": (v / volma).to_numpy(),
                    "ext_atr": ((c - ema20).abs() / atr.replace(0, np.nan)).to_numpy(),
                    "mom1": (c.pct_change() * 100).to_numpy(),
                    "mom5": (c.pct_change(5) * 100).to_numpy(),
                    "mom20": (c.pct_change(20) * 100).to_numpy(),
                    "mom60": (c.pct_change(60) * 100).to_numpy(),
                    "mom120": (c.pct_change(120) * 100).to_numpy(),
                    "break_sig": brk,
                    "dist_52w_high": ((c / hi52 - 1) * 100).to_numpy(),
                    "dist_52w_low": ((c / lo52 - 1) * 100).to_numpy(),
                    "gap_pct": ((o / c.shift(1) - 1) * 100).to_numpy(),
                    "compress": (rng5 / rng20.replace(0, np.nan)).to_numpy(),
                    "month": idx.month,
                }
            ).replace([np.inf, -np.inf], np.nan)
            df = df.merge(reg, on="Date", how="left")
            df["rs20"] = df["mom20"] - df["nifty_mom20"]
            df["rs60"] = df["mom60"] - df["nifty_mom60"]
            df = df.dropna()
            if df.empty:
                continue
            last = df.iloc[-1].copy()
            last["Symbol"] = sym
            last["DTE"] = dte
            last["Expiry"] = expiry.isoformat()
            rows.append(last)
        except Exception:
            continue
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # cross-sectional ranks on latest cross-section
    frame["rank_mom20"] = frame["mom20"].rank(pct=True)
    frame["rank_rs20"] = frame["rs20"].rank(pct=True)
    frame["rank_atr"] = frame["atr_pct"].rank(pct=True)
    frame["rank_vol"] = frame["vol_ratio"].rank(pct=True)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"✅ TwentyPct80 frame: {len(frame)}")
    return frame


def scan(df: pd.DataFrame | None = None) -> pd.DataFrame:
    bundle = load_bundle()
    clf = bundle["model"]
    feats = bundle["features"]
    hold = int(bundle["hold"])
    mode = bundle.get("mode") or "thr"
    k = int(bundle.get("k") or 1)
    thr = bundle.get("thr")
    thr = float(thr) if thr is not None else None
    chosen = bundle.get("chosen") or {}
    frame = df if df is not None else build_live_feature_frame()
    if frame.empty:
        return pd.DataFrame()
    x = frame[feats]
    p = clf.predict_proba(x)[:, 1]
    order = np.argsort(-p)
    if mode == "daily_topk":
        take = order[:k]
    else:
        take = [i for i in order if thr is not None and p[i] >= thr][: max(k, 3)]
    hits = []
    for rank, idx in enumerate(take, start=1):
        r = frame.iloc[int(idx)]
        entry = float(r["Close"])
        atr = float(r["ATR"])
        target = entry * 1.20
        stop = entry - 1.5 * atr
        atm = suggest_atm_strike(entry)
        if mode == "daily_topk":
            reason = (
                f"Daily top-{k} #{rank}: calibrated P(+20% in ≤{hold}d)={p[idx]:.1%}. "
                f"Backtest WR={chosen.get('back_wr', float('nan')):.1%} @ {chosen.get('back_tpd', 1):.1f}/day; "
                f"forward WR={chosen.get('fwd_wr', float('nan')):.1%} (n={chosen.get('fwd_n')}, "
                f"Wilson={chosen.get('wilson', float('nan')):.1%})."
            )
        else:
            reason = (
                f"Calibrated P(+20% in ≤{hold}d)={p[idx]:.1%} ≥ {thr:.0%}. "
                f"Backtest WR={chosen.get('back_wr', float('nan')):.1%}, "
                f"forward WR={chosen.get('fwd_wr', float('nan')):.1%}."
            )
        hits.append(
            {
                "Symbol": r["Symbol"],
                "Bias": "CALL",
                "Scanner": SCANNER_NAME,
                "Rank": rank,
                "Conviction": round(float(p[idx]), 4),
                "Threshold": thr,
                "Mode": mode,
                "HoldDays": hold,
                "Option": f"CALL ATM~{atm:.0f} | target +20% within {hold} sessions",
                "Reason": reason,
                "Entry": round(entry, 2),
                "StopLoss": round(stop, 2),
                "Target20": round(target, 2),
                "ATR": round(atr, 2),
                "RSI": round(float(r["rsi"]), 1),
                "NiftyBull": int(r.get("nifty_bull") or 0),
                "DTE": int(r.get("DTE") or 0),
                "Expiry": r.get("Expiry"),
                "BackWR": chosen.get("back_wr"),
                "FwdWR": chosen.get("fwd_wr"),
                "Wilson": chosen.get("wilson"),
                "TipsPerDay": chosen.get("fwd_tpd") or chosen.get("back_tpd") or 1.0,
            }
        )
    if not hits:
        return pd.DataFrame()
    return pd.DataFrame(hits)


def send_email(hits: pd.DataFrame, bundle: dict) -> None:
    if hits.empty:
        print("No TwentyPct80 setups — suppressing email.")
        return
    sender, password, receiver = require_email_secrets()
    chosen = bundle.get("chosen") or {}
    hold = bundle.get("hold")
    mode = bundle.get("mode") or "thr"
    k = bundle.get("k") or 1
    blocks = []
    for _, r in hits.iterrows():
        thr_txt = f"{r['Threshold']:.0%}" if pd.notna(r.get("Threshold")) else "daily top-1"
        blocks.append(
            f"""
            <div style="border:2px solid #084;padding:12px;margin:10px 0;">
              <h3 style="margin:0 0 6px 0;">#{int(r['Rank'])} {r['Symbol']} — CALL / +20% in ≤{hold}d</h3>
              <p><b>Model P(win):</b> {r['Conviction']:.1%} ({thr_txt})</p>
              <p><b>Validated:</b> backtest WR {(r['BackWR'] or 0)*100:.1f}% ·
                 forward WR {(r['FwdWR'] or 0)*100:.1f}% · Wilson {(r['Wilson'] or 0)*100:.1f}% ·
                 ~{(r['TipsPerDay'] or 1):.1f} tip/day</p>
              <p><b>Why:</b> {r['Reason']}</p>
              <p>Entry {r['Entry']} · Stop {r['StopLoss']} (1.5×ATR risk) · Target {r['Target20']} (+20%) · Hold ≤{hold} sessions</p>
              <p>{r['Option']} · DTE {r['DTE']} · {r['Expiry']} · NiftyBull={r['NiftyBull']}</p>
            </div>
            """
        )
    html = f"""
    <html><body style="font-family:sans-serif;">
      <h2>TwentyPct80 Daily — ≥1 stock/day (+20% / ≤{hold}d, validated ≥80% tip WR)</h2>
      <p>
        Mode: <b>{mode}</b> (top {k} each session). Win = underlying prints
        <b>+20% MFE within {hold} trading days</b> (~{hold/21:.0f} months).
        Forward n={chosen.get('fwd_n')} · Wilson={chosen.get('wilson') and chosen.get('wilson')*100:.0f}% ·
        tips/day={chosen.get('fwd_tpd', 1)}.
        Size modest — long hold budget.
      </p>
      {''.join(blocks)}
      <p><i>See scanners/artifacts/scanner20pct_80wr_report.md</i></p>
    </body></html>
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"TwentyPct80 Daily — {len(hits)} pick(s) (+20% / {hold}d, ≥1/day, ≥80% WR)"
    msg["From"] = sender
    msg["To"] = receiver
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
    print(f"📨 TwentyPct80 mail sent: {len(hits)}")


def main() -> None:
    require_email_secrets()
    bundle = load_bundle()
    hits = scan()
    print(f"🎯 TwentyPct80 hits: {len(hits)} (mode={bundle.get('mode')}, k={bundle.get('k')})")
    if not hits.empty:
        cols = ["Rank", "Symbol", "Conviction", "Entry", "StopLoss", "Target20", "HoldDays", "FwdWR", "Wilson"]
        print(hits[cols].to_string(index=False))
    send_email(hits, bundle)


if __name__ == "__main__":
    main()

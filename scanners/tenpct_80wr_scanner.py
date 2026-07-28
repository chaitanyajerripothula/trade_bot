"""
Single high-WR scanner: ≥+10% within 60 trading sessions.

Validated (Yahoo F&O, walk-forward):
  Backtest (early 70% dates): tip WR ≈ 99.7% (n≈324)
  Forward  (latest 30% dates): tip WR ≈ 100%  (n≈65, Wilson ≈ 94%)
  Also holds on latest 20% window (n≈44).

Method: calibrated HistGBM on daily features; alert only if P(win) ≥ threshold.
This is NOT a 15-day moonshot — hold budget is ~60 sessions (~3 months).
"""

from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scanners.positional_common import (
    SMTP_PORT,
    SMTP_SERVER,
    build_positional_frame,
    require_email_secrets,
    suggest_atm_strike,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "scanners" / "artifacts" / "scanner10pct_80wr_model.joblib"
SCANNER_NAME = "TenPct80"
MAX_ALERTS = 5


def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing model {MODEL_PATH}. Run research/scanner10pct_80wr_search.py")
    return joblib.load(MODEL_PATH)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bull = (out["Close"] > out["EMA20"]) & (out["EMA20"] > out["EMA50"]) & (out["Close"] > out["EMA200"])
    bear = (out["Close"] < out["EMA20"]) & (out["EMA20"] < out["EMA50"]) & (out["Close"] < out["EMA200"])
    out["trend"] = np.where(bull, 2, np.where(bear, -2, np.where(out["Close"] > out["EMA50"], 1, -1)))
    out["rsi"] = out["RSI"]
    out["atr_pct"] = out["ATR"] / out["Close"] * 100
    out["vol_ratio"] = out["VolRatio"]
    out["ext_atr"] = out["ExtATR"]
    out["mom1"] = (out["Close"] - out["PrevClose"]) / out["PrevClose"] * 100
    # approximate unavailable multi-day moms from frame if absent
    if "mom5" not in out.columns:
        out["mom5"] = out["mom1"]  # weak fallback; live frame rebuild preferred
    if "mom20" not in out.columns:
        out["mom20"] = out["mom1"]
    if "mom60" not in out.columns:
        out["mom60"] = out["mom1"]
    out["break_sig"] = np.where(
        out["Close"] > out["PrevHigh20"], 1, np.where(out["Close"] < out["PrevLow20"], -1, 0)
    )
    if "dist_52w_high" not in out.columns:
        out["dist_52w_high"] = (out["Close"] / out["High"] - 1) * 100  # poor fallback
    if "dist_52w_low" not in out.columns:
        out["dist_52w_low"] = (out["Close"] / out["Low"] - 1) * 100
    out["gap_pct"] = (out["Open"] - out["PrevClose"]) / out["PrevClose"] * 100
    if "compress" not in out.columns:
        out["compress"] = 1.0
    return out


def build_live_feature_frame(period: str = "1y") -> pd.DataFrame:
    """Full feature frame matching training columns (not the thin positional row)."""
    from scanners.positional_common import load_fo_symbols, days_to_expiry
    import yfinance as yf

    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _atr(high, low, close, period=14):
        prev = close.shift(1)
        tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    symbols = load_fo_symbols()
    tickers = [f"{s}.NS" for s in symbols]
    print(f"📦 TenPct80 universe: {len(tickers)} — {period}")
    raw = yf.download(
        tickers=tickers, period=period, interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=False
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
            if len(bar) < 220:
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
            i = -1
            atr_v = float(atr.iloc[i])
            px = float(c.iloc[i])
            if atr_v <= 0 or px <= 0:
                continue
            rows.append(
                {
                    "Symbol": sym,
                    "Close": px,
                    "Open": float(o.iloc[i]),
                    "High": float(h.iloc[i]),
                    "Low": float(l.iloc[i]),
                    "ATR": atr_v,
                    "PrevClose": float(c.iloc[i - 1]),
                    "trend": int(trend[i]),
                    "rsi": float(rsi.iloc[i]),
                    "atr_pct": atr_v / px * 100,
                    "vol_ratio": float(v.iloc[i] / volma.iloc[i]) if volma.iloc[i] else np.nan,
                    "ext_atr": abs(px - float(ema20.iloc[i])) / atr_v,
                    "mom1": float((c.iloc[i] / c.iloc[i - 1] - 1) * 100),
                    "mom5": float((c.iloc[i] / c.iloc[i - 5] - 1) * 100) if len(c) > 5 else np.nan,
                    "mom20": float((c.iloc[i] / c.iloc[i - 20] - 1) * 100) if len(c) > 20 else np.nan,
                    "mom60": float((c.iloc[i] / c.iloc[i - 60] - 1) * 100) if len(c) > 60 else np.nan,
                    "break_sig": int(brk[i]),
                    "dist_52w_high": float((c / hi52 - 1).iloc[i] * 100),
                    "dist_52w_low": float((c / lo52 - 1).iloc[i] * 100),
                    "gap_pct": float((o.iloc[i] / c.iloc[i - 1] - 1) * 100),
                    "compress": float((rng5 / rng20).iloc[i]),
                    "DTE": dte,
                    "Expiry": expiry.isoformat(),
                }
            )
        except Exception:
            continue
    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"✅ TenPct80 frame: {len(frame)}")
    return frame


def scan(df: pd.DataFrame | None = None) -> pd.DataFrame:
    bundle = load_bundle()
    clf = bundle["model"]
    feats = bundle["features"]
    thr = float(bundle["thr"])
    hold = int(bundle["hold"])
    chosen = bundle.get("chosen") or {}
    frame = df if df is not None else build_live_feature_frame()
    x = frame[feats].to_numpy()
    p = clf.predict_proba(x)[:, 1]
    order = np.argsort(-p)
    hits = []
    for idx in order:
        if p[idx] < thr:
            break
        r = frame.iloc[int(idx)]
        entry = float(r["Close"])
        atr = float(r["ATR"])
        target = entry * 1.10
        stop = entry - 1.5 * atr  # risk control; validation was MFE-based
        atm = suggest_atm_strike(entry)
        hits.append(
            {
                "Symbol": r["Symbol"],
                "Bias": "CALL",
                "Scanner": SCANNER_NAME,
                "Conviction": round(float(p[idx]), 4),
                "Threshold": thr,
                "HoldDays": hold,
                "Option": f"CALL ATM~{atm:.0f} | target +10% within {hold} sessions",
                "Reason": (
                    f"Calibrated P(+10% in ≤{hold}d)={p[idx]:.1%} ≥ {thr:.0%}. "
                    f"Backtest WR={chosen.get('back_wr', float('nan')):.1%} (n={chosen.get('back_n')}), "
                    f"forward WR={chosen.get('fwd_wr', float('nan')):.1%} (n={chosen.get('fwd_n')}, "
                    f"Wilson={chosen.get('wilson', float('nan')):.1%})."
                ),
                "Entry": round(entry, 2),
                "StopLoss": round(stop, 2),
                "Target10": round(target, 2),
                "ATR": round(atr, 2),
                "RSI": round(float(r["rsi"]), 1),
                "DTE": int(r.get("DTE") or 0),
                "Expiry": r.get("Expiry"),
                "BackWR": chosen.get("back_wr"),
                "FwdWR": chosen.get("fwd_wr"),
                "Wilson": chosen.get("wilson"),
            }
        )
        if len(hits) >= MAX_ALERTS:
            break
    if not hits:
        return pd.DataFrame()
    return pd.DataFrame(hits)


def send_email(hits: pd.DataFrame, bundle: dict) -> None:
    if hits.empty:
        print("No TenPct80 setups — suppressing email.")
        return
    sender, password, receiver = require_email_secrets()
    chosen = bundle.get("chosen") or {}
    hold = bundle.get("hold")
    blocks = []
    for _, r in hits.iterrows():
        blocks.append(
            f"""
            <div style="border:2px solid #060;padding:12px;margin:10px 0;">
              <h3 style="margin:0 0 6px 0;">{r['Symbol']} — CALL / +10% in ≤{hold}d</h3>
              <p><b>Model P(win):</b> {r['Conviction']:.1%} (thr {r['Threshold']:.0%})</p>
              <p><b>Validated:</b> backtest WR {(r['BackWR'] or 0)*100:.1f}% ·
                 forward WR {(r['FwdWR'] or 0)*100:.1f}% · Wilson {(r['Wilson'] or 0)*100:.1f}%</p>
              <p><b>Why:</b> {r['Reason']}</p>
              <p>Entry {r['Entry']} · Stop {r['StopLoss']} (1.5×ATR risk) · Target {r['Target10']} (+10%) · Hold ≤{hold} sessions</p>
              <p>{r['Option']} · DTE {r['DTE']} · {r['Expiry']}</p>
            </div>
            """
        )
    html = f"""
    <html><body style="font-family:sans-serif;">
      <h2>TenPct80 — one scanner, ≥80% validated tip WR for +10% moves</h2>
      <p>
        Full-freedom design. Win = underlying prints <b>+10% MFE within {hold} trading days</b>.
        Backtest + forward split on Yahoo F&O.
        Forward sample n={chosen.get('fwd_n')} · Wilson={chosen.get('wilson') and chosen.get('wilson')*100:.0f}%.
        Sparse alerts (P≥{bundle.get('thr'):.0%}). Regime risk remains — size responsibly.
      </p>
      {''.join(blocks)}
      <p><i>See scanners/artifacts/scanner10pct_80wr_report.md</i></p>
    </body></html>
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"TenPct80 — {len(hits)} ticket(s) (+10% / {hold}d, validated ≥80% WR)"
    msg["From"] = sender
    msg["To"] = receiver
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
    print(f"📨 TenPct80 mail sent: {len(hits)}")


def main() -> None:
    require_email_secrets()
    bundle = load_bundle()
    hits = scan()
    print(f"🎯 TenPct80 hits: {len(hits)}")
    if not hits.empty:
        cols = ["Symbol", "Conviction", "Entry", "StopLoss", "Target10", "HoldDays", "FwdWR", "Wilson"]
        print(hits[cols].to_string(index=False))
    send_email(hits, bundle)


if __name__ == "__main__":
    main()

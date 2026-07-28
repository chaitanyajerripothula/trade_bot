"""
Short-hold positional scanner (max 1–2 months).

User constraints: hold ≤ ~30–45 trading days, prefer ≥1 tip/day, prefer ≥80% tip WR.

Under those caps:
  - +20% @ ≥80% WR is NOT attainable (best ~58% @45d / ~68% @60d with ≥1 avg tip/day)
  - Forcing a tip *every* calendar day also breaks 80% (daily top-1 +10% ≈ 57–61% fwd WR)

Shipped rule (best attainable):
  Win = +10% MFE within **30** trading days (~1.5 months)
  Alert if calibrated P ≥ **0.82** (max 3 tips)
  Backtest WR ≈ 96.7% · Forward WR ≈ 88.7% (Wilson ≈ 85%)
  Average ≈ 1.4 tips/day; tips cluster (~17% of days fire) — quiet days are expected
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
MODEL_CANDIDATES = [
    ROOT / "scanners" / "artifacts" / "scanner_short_hold_80_model.joblib",
    ROOT / "scanners" / "artifacts" / "scanner_short_hold_1perday_model.joblib",
    ROOT / "scanners" / "artifacts" / "scanner20pct_80wr_model.joblib",
]
SCANNER_NAME = "ShortHold80"


def load_bundle() -> dict:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return joblib.load(path)
    raise SystemExit(f"Missing ShortHold80 model. Tried: {', '.join(str(p) for p in MODEL_CANDIDATES)}")


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
    print(f"📦 {SCANNER_NAME} universe: {len(tickers)} — {period}")
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
    frame["rank_mom20"] = frame["mom20"].rank(pct=True)
    frame["rank_rs20"] = frame["rs20"].rank(pct=True)
    frame["rank_atr"] = frame["atr_pct"].rank(pct=True)
    frame["rank_vol"] = frame["vol_ratio"].rank(pct=True)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"✅ {SCANNER_NAME} frame: {len(frame)}")
    return frame


def scan(df: pd.DataFrame | None = None) -> pd.DataFrame:
    bundle = load_bundle()
    clf = bundle["model"]
    feats = bundle["features"]
    hold = int(bundle["hold"])
    target_pct = float(bundle.get("target") or 0.10)
    mode = bundle.get("mode") or "thr"
    max_k = int(bundle.get("max_k") or bundle.get("k") or 3)
    thr = bundle.get("thr")
    thr = float(thr) if thr is not None else None
    chosen = bundle.get("chosen") or {}
    frame = df if df is not None else build_live_feature_frame()
    if frame.empty:
        return pd.DataFrame()
    p = clf.predict_proba(frame[feats])[:, 1]
    order = np.argsort(-p)
    if mode == "daily_topk":
        take = order[:max_k]
    else:
        take = [i for i in order if thr is not None and p[i] >= thr][:max_k]
    hits = []
    tgt_label = f"+{target_pct*100:.0f}%"
    for rank, idx in enumerate(take, start=1):
        r = frame.iloc[int(idx)]
        entry = float(r["Close"])
        atr = float(r["ATR"])
        target_px = entry * (1.0 + target_pct)
        stop = entry - 1.5 * atr
        atm = suggest_atm_strike(entry)
        reason = (
            f"Calibrated P({tgt_label} in ≤{hold}d)={p[idx]:.1%}"
            + (f" ≥ {thr:.0%}." if thr is not None else ".")
            + f" Backtest WR={chosen.get('back_wr', float('nan')):.1%} · "
            f"forward WR={chosen.get('fwd_wr', float('nan')):.1%} "
            f"(n={chosen.get('fwd_n')}, Wilson={chosen.get('wilson', float('nan')):.1%}) · "
            f"avg ~{chosen.get('fwd_tpd', float('nan')):.1f} tips/day."
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
                "TargetPct": target_pct,
                "Option": f"CALL ATM~{atm:.0f} | target {tgt_label} within {hold} sessions",
                "Reason": reason,
                "Entry": round(entry, 2),
                "StopLoss": round(stop, 2),
                "TargetPx": round(target_px, 2),
                "ATR": round(atr, 2),
                "RSI": round(float(r["rsi"]), 1),
                "NiftyBull": int(r.get("nifty_bull") or 0),
                "DTE": int(r.get("DTE") or 0),
                "Expiry": r.get("Expiry"),
                "BackWR": chosen.get("back_wr"),
                "FwdWR": chosen.get("fwd_wr"),
                "Wilson": chosen.get("wilson"),
                "TipsPerDay": chosen.get("fwd_tpd") or chosen.get("back_tpd"),
            }
        )
    if not hits:
        return pd.DataFrame()
    return pd.DataFrame(hits)


def send_email(hits: pd.DataFrame, bundle: dict) -> None:
    if hits.empty:
        print(f"No {SCANNER_NAME} setups — quiet day (expected; tips cluster on high-conviction sessions).")
        return
    sender, password, receiver = require_email_secrets()
    chosen = bundle.get("chosen") or {}
    hold = bundle.get("hold")
    target_pct = float(bundle.get("target") or 0.10)
    tgt_label = f"+{target_pct*100:.0f}%"
    blocks = []
    for _, r in hits.iterrows():
        thr_txt = f"{r['Threshold']:.0%}" if pd.notna(r.get("Threshold")) else "n/a"
        blocks.append(
            f"""
            <div style="border:2px solid #084;padding:12px;margin:10px 0;">
              <h3 style="margin:0 0 6px 0;">#{int(r['Rank'])} {r['Symbol']} — CALL / {tgt_label} in ≤{hold}d</h3>
              <p><b>Model P(win):</b> {r['Conviction']:.1%} (thr {thr_txt})</p>
              <p><b>Validated:</b> backtest WR {(r['BackWR'] or 0)*100:.1f}% ·
                 forward WR {(r['FwdWR'] or 0)*100:.1f}% · Wilson {(r['Wilson'] or 0)*100:.1f}% ·
                 ~{(r['TipsPerDay'] or 0):.1f} tips/day avg</p>
              <p><b>Why:</b> {r['Reason']}</p>
              <p>Entry {r['Entry']} · Stop {r['StopLoss']} (1.5×ATR) · Target {r['TargetPx']} ({tgt_label}) · Hold ≤{hold} sessions (~1.5 months)</p>
              <p>{r['Option']} · DTE {r['DTE']} · {r['Expiry']}</p>
            </div>
            """
        )
    html = f"""
    <html><body style="font-family:sans-serif;">
      <h2>{SCANNER_NAME} — {tgt_label} / ≤{hold}d (max hold 1–2 months, ≥80% tip WR)</h2>
      <p>
        Win = underlying prints <b>{tgt_label} MFE within {hold} trading days</b>.
        Forward WR={(chosen.get('fwd_wr') or 0)*100:.1f}% (n={chosen.get('fwd_n')}, Wilson={(chosen.get('wilson') or 0)*100:.0f}%).
        Avg tips/day≈{(chosen.get('fwd_tpd') or 0):.1f}. Quiet days are normal — forcing a tip every day drops WR below 80%.
        <b>+20% at 80% WR is not attainable inside a 1–2 month hold with this frequency.</b>
      </p>
      {''.join(blocks)}
      <p><i>See scanners/artifacts/scanner20pct_80wr_report.md</i></p>
    </body></html>
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"{SCANNER_NAME} — {len(hits)} pick(s) ({tgt_label} / {hold}d, ≥80% WR)"
    msg["From"] = sender
    msg["To"] = receiver
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
    print(f"📨 {SCANNER_NAME} mail sent: {len(hits)}")


def main(dry_run: bool = False) -> pd.DataFrame:
    if not dry_run:
        require_email_secrets()
    bundle = load_bundle()
    hits = scan()
    tgt = float(bundle.get("target") or 0.10)
    print(
        f"🎯 {SCANNER_NAME} hits: {len(hits)} "
        f"(target=+{tgt*100:.0f}% hold={bundle.get('hold')}d thr={bundle.get('thr')} dry_run={dry_run})"
    )
    if not hits.empty:
        cols = ["Rank", "Symbol", "Conviction", "Entry", "StopLoss", "TargetPx", "HoldDays", "FwdWR", "Wilson"]
        print(hits[cols].to_string(index=False))
    else:
        print("Quiet day — no name cleared the ≥80% WR threshold.")
    if dry_run:
        print("Dry-run: email suppressed.")
    else:
        send_email(hits, bundle)
    return hits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ShortHold80: +10% in ≤30d @ ≥80% tip WR")
    parser.add_argument("--dry-run", action="store_true", help="Scan and print only; skip email")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
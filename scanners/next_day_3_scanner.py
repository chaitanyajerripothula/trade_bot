"""
NextDay3 — next-session ≥3% move from open (either direction).

Win: max(open→high, open→low) ≥ 3% on the *next* trading day.
Tip at EOD. Bias=MOVE (direction is not predictable at ≥80% WR).

Validation (walk-forward): ~93% back / ~85% forward (Wilson ~71%),
P≥0.82 max 3 tips · ~0.14 tips/day ≈ 3/month.

Strict unidirectional and directional CALL/PUT variants did NOT clear 80%.
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
)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "scanners" / "artifacts"
MODEL_PATH = ART / "next_day_3_model.joblib"
HISTORY_PATH = ART / "next_day_3_signal_history.json"
SCANNER_NAME = "NextDay3"
HISTORY_KEEP = 50
LAST_N = 10
HISTORY_COLS = ["Date", "Symbol", "Bias", "Entry", "Exit", "StopLoss", "Conviction", "Reason"]


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


def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing model {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=HISTORY_COLS)
    raw = pd.read_json(HISTORY_PATH)
    if raw.empty:
        return pd.DataFrame(columns=HISTORY_COLS)
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.strftime("%Y-%m-%d")
    for col in HISTORY_COLS:
        if col not in raw.columns:
            raw[col] = np.nan
    return raw[HISTORY_COLS]


def save_history(df: pd.DataFrame) -> None:
    out = df.sort_values(["Date", "Conviction"], ascending=[False, False]).head(HISTORY_KEEP)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_json(HISTORY_PATH, orient="records", indent=2, date_format="iso")


def append_signals(hits: pd.DataFrame) -> pd.DataFrame:
    hist = load_history()
    if hits is None or hits.empty:
        return hist.sort_values(["Date", "Conviction"], ascending=[False, False]).reset_index(drop=True)
    day = pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%d")
    rows = []
    for _, r in hits.iterrows():
        rows.append(
            {
                "Date": day,
                "Symbol": r["Symbol"],
                "Bias": r.get("Bias", "MOVE"),
                "Entry": float(r["Entry"]),
                "Exit": float(r["Exit"]),
                "StopLoss": float(r["StopLoss"]),
                "Conviction": float(r["Conviction"]),
                "Reason": r["Reason"],
            }
        )
    combined = pd.concat([hist, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "Symbol"], keep="last")
    combined = combined.sort_values(["Date", "Conviction"], ascending=[False, False]).head(HISTORY_KEEP)
    save_history(combined)
    return combined.reset_index(drop=True)


def last_signals(n: int = LAST_N) -> pd.DataFrame:
    hist = load_history()
    if hist.empty:
        return hist
    return hist.sort_values(["Date", "Conviction"], ascending=[False, False]).head(n).reset_index(drop=True)


def backfill_signal_history(min_n: int = LAST_N, lookback_days: int = 250) -> pd.DataFrame:
    """Replay recent sessions until ≥min_n historical tips (sparse ~3/month)."""
    hist = load_history()
    if len(hist) >= min_n:
        return last_signals(min_n)

    bundle = load_bundle()
    clf, feats = bundle["model"], bundle["features"]
    thr = float(bundle.get("thr") or 0.82)
    max_k = int(bundle.get("max_k") or 3)
    move = float(bundle.get("move") or 0.03)
    chosen = bundle.get("chosen") or {}

    symbols = load_fo_symbols()
    tickers = [f"{s}.NS" for s in symbols]
    print(f"⏪ Backfilling {SCANNER_NAME} history from last ~{lookback_days}d…")
    raw = yf.download(
        tickers=tickers, period="2y", interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=False
    )
    reg = _nifty_regime()
    panels = {}
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
            volstd = v.rolling(20).std()
            prev_hi = h.shift(1).rolling(20).max()
            prev_lo = l.shift(1).rolling(20).min()
            hi52 = h.rolling(252).max()
            lo52 = l.rolling(252).min()
            rng20 = (h.rolling(20).max() - l.rolling(20).min()) / c.replace(0, np.nan)
            rng5 = (h.rolling(5).max() - l.rolling(5).min()) / c.replace(0, np.nan)
            day_rng = (h - l).replace(0, np.nan)
            oc_max = pd.concat([o, c], axis=1).max(axis=1)
            oc_min = pd.concat([o, c], axis=1).min(axis=1)
            bull = (c > ema20) & (ema20 > ema50) & (c > ema200)
            bear = (c < ema20) & (ema20 < ema50) & (c < ema200)
            trend = np.where(bull, 2, np.where(bear, -2, np.where(c > ema50, 1, np.where(c < ema50, -1, 0))))
            brk = np.where(c > prev_hi, 1, np.where(c < prev_lo, -1, 0))
            idx = pd.to_datetime(bar.index).tz_localize(None)
            df = pd.DataFrame(
                {
                    "Date": idx,
                    "Close": c.to_numpy(),
                    "ATR": atr.to_numpy(),
                    "trend": trend,
                    "rsi": rsi.to_numpy(),
                    "atr_pct": (atr / c * 100).to_numpy(),
                    "vol_ratio": (v / volma).to_numpy(),
                    "vol_z": ((v - volma) / volstd.replace(0, np.nan)).to_numpy(),
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
                    "close_loc": ((c - l) / day_rng).to_numpy(),
                    "day_range_pct": ((h - l) / c * 100).to_numpy(),
                    "body_pct": ((c - o).abs() / c * 100).to_numpy(),
                    "upper_wick": ((h - oc_max) / day_rng).to_numpy(),
                    "lower_wick": ((oc_min - l) / day_rng).to_numpy(),
                    "month": idx.month,
                }
            ).replace([np.inf, -np.inf], np.nan)
            df = df.merge(reg, on="Date", how="left")
            df["rs20"] = df["mom20"] - df["nifty_mom20"]
            df["rs60"] = df["mom60"] - df["nifty_mom60"]
            df = df.dropna().reset_index(drop=True)
            if len(df) >= 180:
                panels[sym] = df
        except Exception:
            continue

    if not panels:
        print("Backfill failed — empty panels")
        return last_signals(min_n)

    all_dates = sorted({d for p in panels.values() for d in p["Date"].tolist()})
    recent = all_dates[-lookback_days:]
    collected = []
    for dt in reversed(recent):
        rows = []
        for sym, panel in panels.items():
            sub = panel[panel["Date"] == dt]
            if sub.empty:
                continue
            row = sub.iloc[0].to_dict()
            row["Symbol"] = sym
            rows.append(row)
        if len(rows) < 30:
            continue
        day = pd.DataFrame(rows)
        day["rank_mom20"] = day["mom20"].rank(pct=True)
        day["rank_rs20"] = day["rs20"].rank(pct=True)
        day["rank_atr"] = day["atr_pct"].rank(pct=True)
        day["rank_vol"] = day["vol_ratio"].rank(pct=True)
        day = day.replace([np.inf, -np.inf], np.nan).dropna(subset=feats)
        if day.empty:
            continue
        p = clf.predict_proba(day[feats].to_numpy())[:, 1]
        order = np.argsort(-p)
        take = [i for i in order if p[i] >= thr][:max_k]
        for idx in take:
            r = day.iloc[int(idx)]
            ref = float(r["Close"])
            atr = float(r["ATR"])
            reason = (
                f"P(≥{move*100:.0f}% from next open, either way)={p[idx]:.1%} ≥ {thr:.0%}. "
                f"Forward WR={chosen.get('fwd_wr', float('nan')):.1%} "
                f"(Wilson={chosen.get('wilson', float('nan')):.1%})."
            )
            collected.append(
                {
                    "Date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "Symbol": r["Symbol"],
                    "Bias": "MOVE",
                    "Entry": round(ref, 2),
                    "Exit": round(ref * (1.0 + move), 2),
                    "StopLoss": round(ref - 1.5 * atr, 2),
                    "Conviction": round(float(p[idx]), 4),
                    "Reason": reason,
                }
            )
        if len(collected) >= min_n:
            break

    if collected:
        add = pd.DataFrame(collected)
        combined = pd.concat([hist, add], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date", "Symbol"], keep="last")
        combined = combined.sort_values(["Date", "Conviction"], ascending=[False, False]).head(HISTORY_KEEP)
        save_history(combined)
        print(f"✅ Backfilled {len(add)} historical tip(s); history size={len(combined)}")
    else:
        print("Backfill found no tips above threshold in lookback window")
    return last_signals(min_n)


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
            if len(bar) < 200:
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
            volstd = v.rolling(20).std()
            prev_hi = h.shift(1).rolling(20).max()
            prev_lo = l.shift(1).rolling(20).min()
            hi52 = h.rolling(252).max()
            lo52 = l.rolling(252).min()
            rng20 = (h.rolling(20).max() - l.rolling(20).min()) / c.replace(0, np.nan)
            rng5 = (h.rolling(5).max() - l.rolling(5).min()) / c.replace(0, np.nan)
            day_rng = (h - l).replace(0, np.nan)
            oc_max = pd.concat([o, c], axis=1).max(axis=1)
            oc_min = pd.concat([o, c], axis=1).min(axis=1)
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
                    "vol_z": ((v - volma) / volstd.replace(0, np.nan)).to_numpy(),
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
                    "close_loc": ((c - l) / day_rng).to_numpy(),
                    "day_range_pct": ((h - l) / c * 100).to_numpy(),
                    "body_pct": ((c - o).abs() / c * 100).to_numpy(),
                    "upper_wick": ((h - oc_max) / day_rng).to_numpy(),
                    "lower_wick": ((oc_min - l) / day_rng).to_numpy(),
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


def scan(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    b = load_bundle()
    clf, feats = b["model"], b["features"]
    thr = float(b["thr"])
    max_k = int(b.get("max_k") or 3)
    move = float(b.get("move") or 0.03)
    chosen = b.get("chosen") or {}
    frame = frame if frame is not None else build_live_feature_frame()
    if frame.empty:
        return pd.DataFrame()
    p = clf.predict_proba(frame[feats].to_numpy())[:, 1]
    order = np.argsort(-p)
    take = [i for i in order if p[i] >= thr][:max_k]
    hits = []
    for rank, idx in enumerate(take, start=1):
        r = frame.iloc[int(idx)]
        ref = float(r["Close"])  # reference; true entry = next open
        atr = float(r["ATR"])
        # Exit shown as +3% from ref (actual targets are ±3% from *next open*)
        exit_up = ref * (1.0 + move)
        exit_dn = ref * (1.0 - move)
        reason = (
            f"P(≥{move*100:.0f}% from next open, either way)={p[idx]:.1%} ≥ {thr:.0%}. "
            f"Forward WR={chosen.get('fwd_wr', float('nan')):.1%} "
            f"(Wilson={chosen.get('wilson', float('nan')):.1%}). "
            f"Watch next open → target Open×{1+move:.2f} or Open×{1-move:.2f}."
        )
        hits.append(
            {
                "Symbol": r["Symbol"],
                "Bias": "MOVE",
                "Scanner": SCANNER_NAME,
                "Rank": rank,
                "Conviction": round(float(p[idx]), 4),
                "Threshold": thr,
                "MovePct": move,
                "Option": f"Straddle/watch ±{move*100:.0f}% from next open (direction unknown @80% bar)",
                "Reason": reason,
                "Entry": round(ref, 2),  # ref close; trade from next open
                "Exit": round(exit_up, 2),  # upside target vs ref; downside also valid
                "ExitDown": round(exit_dn, 2),
                "StopLoss": round(ref - 1.5 * atr, 2),
                "ATR": round(atr, 2),
                "RSI": round(float(r["rsi"]), 1),
                "DTE": int(r.get("DTE") or 0),
                "Expiry": r.get("Expiry"),
                "BackWR": chosen.get("back_wr"),
                "FwdWR": chosen.get("fwd_wr"),
                "Wilson": chosen.get("wilson"),
                "TipsPerDay": chosen.get("fwd_tpd"),
            }
        )
    return pd.DataFrame(hits) if hits else pd.DataFrame()


def _history_html(last: pd.DataFrame) -> str:
    if last is None or last.empty:
        return "<p><i>No prior signals in history.</i></p>"
    rows = []
    for _, r in last.iterrows():
        rows.append(
            f"<tr><td>{r['Date']}</td><td><b>{r['Symbol']}</b></td><td>{r['Bias']}</td>"
            f"<td>{float(r['Entry']):.2f}</td><td>{float(r['Exit']):.2f}</td>"
            f"<td style='max-width:420px'>{r['Reason']}</td></tr>"
        )
    return f"""
    <h3>Last {len(last)} signals</h3>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;font-size:13px;">
      <tr><th>Date</th><th>Symbol</th><th>Bias</th><th>Entry(ref)</th><th>Exit(+3% ref)</th><th>Reason</th></tr>
      {''.join(rows)}
    </table>
    """


def send_email(hits: pd.DataFrame, last: pd.DataFrame) -> None:
    sender, password, receiver = require_email_secrets()
    b = load_bundle()
    chosen = b.get("chosen") or {}
    move = float(b.get("move") or 0.03)
    n_new = 0 if hits is None or hits.empty else len(hits)
    blocks = []
    if n_new:
        for _, r in hits.iterrows():
            blocks.append(
                f"""
                <div style="border:2px solid #084;padding:12px;margin:10px 0;">
                  <h3 style="margin:0;">{r['Symbol']} — MOVE ±{move*100:.0f}% from next open</h3>
                  <p>P(win)={r['Conviction']:.1%} · Ref close {r['Entry']} ·
                     Targets ~{r['Exit']} / {r.get('ExitDown', '—')} · Stop(ref) {r['StopLoss']}</p>
                  <p>{r['Reason']}</p>
                </div>
                """
            )
    else:
        blocks.append("<p><b>No new tips today</b> (quiet day — sparse scanner ~3 tips/month).</p>")
    html = f"""
    <html><body style="font-family:sans-serif;">
      <h2>{SCANNER_NAME} — next day ≥{move*100:.0f}% from open (either way)</h2>
      <p>Forward WR={(chosen.get('fwd_wr') or 0)*100:.1f}% · ~{(chosen.get('fwd_tpd') or 0):.2f} tips/day
         · ~{(chosen.get('approx_tips_month') or 0):.0f}/month</p>
      <p><i>Bias=MOVE: direction is not part of the 80% edge. Trade the open excursion either side.</i></p>
      <h3>Today ({n_new} new)</h3>
      {''.join(blocks)}
      {_history_html(last)}
    </body></html>
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"{SCANNER_NAME} — {n_new} new · last {len(last)} signals"
    msg["From"] = sender
    msg["To"] = receiver
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
    print(f"📨 {SCANNER_NAME} mail sent (new={n_new}, history={len(last)})")


def main(dry_run: bool = False) -> pd.DataFrame:
    if not dry_run:
        require_email_secrets()
    if len(load_history()) < LAST_N:
        backfill_signal_history(LAST_N)
    hits = scan()
    hist = append_signals(hits)
    last = hist.head(LAST_N)
    b = load_bundle()
    print(
        f"🎯 {SCANNER_NAME} hits={len(hits)} "
        f"(±{b.get('move', 0.03)*100:.0f}% from next open thr={b['thr']:.2f} dry_run={dry_run})"
    )
    if not hits.empty:
        cols = ["Rank", "Symbol", "Bias", "Entry", "Exit", "ExitDown", "Conviction", "Reason"]
        print(hits[[c for c in cols if c in hits.columns]].to_string(index=False))
    else:
        print("Quiet day.")
    if not last.empty:
        show = last.copy()
        show["Reason"] = show["Reason"].astype(str).str.slice(0, 80)
        print(f"\n📋 Last {len(last)} signals:")
        print(show[["Date", "Symbol", "Bias", "Entry", "Exit", "Reason"]].to_string(index=False))
    if dry_run:
        print("Dry-run: email suppressed.")
    else:
        send_email(hits, last)
    return hits


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    main(dry_run=args.dry_run)

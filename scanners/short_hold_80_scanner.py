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
]
HISTORY_PATH = ROOT / "scanners" / "artifacts" / "short_hold_80_signal_history.json"
SCANNER_NAME = "ShortHold80"
HISTORY_KEEP = 50
LAST_N = 10
HISTORY_COLS = ["Date", "Symbol", "Bias", "Entry", "Exit", "StopLoss", "Conviction", "Reason"]


def load_bundle() -> dict:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return joblib.load(path)
    raise SystemExit(f"Missing ShortHold80 model. Tried: {', '.join(str(p) for p in MODEL_CANDIDATES)}")


def load_signal_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=HISTORY_COLS)
    raw = pd.read_json(HISTORY_PATH)
    if raw.empty:
        return pd.DataFrame(columns=HISTORY_COLS)
    if "Date" in raw.columns:
        raw["Date"] = pd.to_datetime(raw["Date"]).dt.strftime("%Y-%m-%d")
    for col in HISTORY_COLS:
        if col not in raw.columns:
            raw[col] = np.nan
    return raw[HISTORY_COLS]


def save_signal_history(df: pd.DataFrame) -> None:
    out = df.copy()
    if out.empty:
        HISTORY_PATH.write_text("[]")
        return
    out = out.sort_values(["Date", "Conviction"], ascending=[False, False]).head(HISTORY_KEEP)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_json(HISTORY_PATH, orient="records", indent=2, date_format="iso")


def append_signals(hits: pd.DataFrame, as_of: str | None = None) -> pd.DataFrame:
    """Append today's hits to history; return updated history (newest first)."""
    hist = load_signal_history()
    if hits is None or hits.empty:
        return hist.sort_values(["Date", "Conviction"], ascending=[False, False]).reset_index(drop=True)
    day = as_of or pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%d")
    rows = []
    for _, r in hits.iterrows():
        rows.append(
            {
                "Date": day,
                "Symbol": r["Symbol"],
                "Bias": r.get("Bias", "CALL"),
                "Entry": float(r["Entry"]),
                "Exit": float(r.get("Exit", r.get("TargetPx"))),
                "StopLoss": float(r["StopLoss"]),
                "Conviction": float(r["Conviction"]),
                "Reason": r["Reason"],
            }
        )
    add = pd.DataFrame(rows)
    # de-dupe same day+symbol
    combined = pd.concat([hist, add], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "Symbol"], keep="last")
    combined = combined.sort_values(["Date", "Conviction"], ascending=[False, False]).head(HISTORY_KEEP)
    save_signal_history(combined)
    return combined.reset_index(drop=True)


def last_signals(n: int = LAST_N) -> pd.DataFrame:
    hist = load_signal_history()
    if hist.empty:
        return hist
    return hist.sort_values(["Date", "Conviction"], ascending=[False, False]).head(n).reset_index(drop=True)


def format_last_signals_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "(no signals in history yet)"
    show = df.copy()
    show["Entry"] = show["Entry"].map(lambda x: f"{float(x):.2f}")
    show["Exit"] = show["Exit"].map(lambda x: f"{float(x):.2f}")
    show["StopLoss"] = show["StopLoss"].map(lambda x: f"{float(x):.2f}")
    show["Conviction"] = show["Conviction"].map(lambda x: f"{float(x):.1%}")
    # truncate reason for CLI
    show["Reason"] = show["Reason"].astype(str).str.slice(0, 90)
    return show[["Date", "Symbol", "Bias", "Entry", "Exit", "Reason"]].to_string(index=False)


def backfill_signal_history(min_n: int = LAST_N, lookback_days: int = 90) -> pd.DataFrame:
    """Replay recent sessions with the live model until we have ≥min_n historical tips."""
    hist = load_signal_history()
    if len(hist) >= min_n:
        return last_signals(min_n)

    bundle = load_bundle()
    clf = bundle["model"]
    feats = bundle["features"]
    hold = int(bundle["hold"])
    target_pct = float(bundle.get("target") or 0.10)
    thr = float(bundle.get("thr") or 0.82)
    max_k = int(bundle.get("max_k") or 3)
    chosen = bundle.get("chosen") or {}

    symbols = load_fo_symbols()
    tickers = [f"{s}.NS" for s in symbols]
    print(f"⏪ Backfilling ShortHold80 history from last ~{lookback_days}d…")
    raw = yf.download(
        tickers=tickers, period="2y", interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=False
    )
    reg = _nifty_regime()
    # Build per-symbol feature panels
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
    tgt_label = f"+{target_pct*100:.0f}%"
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
        p = clf.predict_proba(day[feats])[:, 1]
        order = np.argsort(-p)
        take = [i for i in order if p[i] >= thr][:max_k]
        for idx in take:
            r = day.iloc[int(idx)]
            entry = float(r["Close"])
            atr = float(r["ATR"])
            exit_px = entry * (1.0 + target_pct)
            stop = entry - 1.5 * atr
            reason = (
                f"Calibrated P({tgt_label} in ≤{hold}d)={p[idx]:.1%} ≥ {thr:.0%}. "
                f"Forward WR={chosen.get('fwd_wr', float('nan')):.1%} "
                f"(Wilson={chosen.get('wilson', float('nan')):.1%})."
            )
            collected.append(
                {
                    "Date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "Symbol": r["Symbol"],
                    "Bias": "CALL",
                    "Entry": round(entry, 2),
                    "Exit": round(exit_px, 2),
                    "StopLoss": round(stop, 2),
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
        save_signal_history(combined)
        print(f"✅ Backfilled {len(add)} historical tip(s); history size={len(combined)}")
    else:
        print("Backfill found no tips above threshold in lookback window")
    return last_signals(min_n)


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
                "Exit": round(target_px, 2),
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


def _history_html(last: pd.DataFrame) -> str:
    if last is None or last.empty:
        return "<p><i>No prior signals in history.</i></p>"
    rows = []
    for _, r in last.iterrows():
        rows.append(
            "<tr>"
            f"<td>{r['Date']}</td>"
            f"<td><b>{r['Symbol']}</b></td>"
            f"<td>{r['Bias']}</td>"
            f"<td>{float(r['Entry']):.2f}</td>"
            f"<td>{float(r['Exit']):.2f}</td>"
            f"<td style='max-width:420px'>{r['Reason']}</td>"
            "</tr>"
        )
    return f"""
    <h3>Last {len(last)} signals</h3>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;font-size:13px;">
      <tr>
        <th>Date</th><th>Symbol</th><th>Bias</th><th>Entry</th><th>Exit (+10% target)</th><th>Reason</th>
      </tr>
      {''.join(rows)}
    </table>
    """


def send_email(hits: pd.DataFrame, bundle: dict, last: pd.DataFrame | None = None) -> None:
    sender, password, receiver = require_email_secrets()
    chosen = bundle.get("chosen") or {}
    hold = bundle.get("hold")
    target_pct = float(bundle.get("target") or 0.10)
    tgt_label = f"+{target_pct*100:.0f}%"
    last = last if last is not None else last_signals(LAST_N)
    blocks = []
    if hits is not None and not hits.empty:
        for _, r in hits.iterrows():
            thr_txt = f"{r['Threshold']:.0%}" if pd.notna(r.get("Threshold")) else "n/a"
            blocks.append(
                f"""
                <div style="border:2px solid #084;padding:12px;margin:10px 0;">
                  <h3 style="margin:0 0 6px 0;">#{int(r['Rank'])} {r['Symbol']} — {r['Bias']} / {tgt_label} in ≤{hold}d</h3>
                  <p><b>Model P(win):</b> {r['Conviction']:.1%} (thr {thr_txt})</p>
                  <p><b>Validated:</b> backtest WR {(r['BackWR'] or 0)*100:.1f}% ·
                     forward WR {(r['FwdWR'] or 0)*100:.1f}% · Wilson {(r['Wilson'] or 0)*100:.1f}% ·
                     ~{(r['TipsPerDay'] or 0):.1f} tips/day avg</p>
                  <p><b>Why:</b> {r['Reason']}</p>
                  <p>Entry {r['Entry']} · Exit {r['Exit']} ({tgt_label}) · Stop {r['StopLoss']} (1.5×ATR) · Hold ≤{hold} sessions</p>
                  <p>{r['Option']} · DTE {r['DTE']} · {r['Expiry']}</p>
                </div>
                """
            )
    else:
        blocks.append("<p><b>No new tips today</b> (quiet day — expected).</p>")

    n_new = 0 if hits is None or hits.empty else len(hits)
    html = f"""
    <html><body style="font-family:sans-serif;">
      <h2>{SCANNER_NAME} — {tgt_label} / ≤{hold}d (≥80% tip WR)</h2>
      <p>
        Win = underlying prints <b>{tgt_label} MFE within {hold} trading days</b>.
        Forward WR={(chosen.get('fwd_wr') or 0)*100:.1f}% · avg tips/day≈{(chosen.get('fwd_tpd') or 0):.1f}.
        Exit = planned +10% target price (not a filled trade).
      </p>
      <h3>Today ({n_new} new)</h3>
      {''.join(blocks)}
      {_history_html(last)}
      <p><i>History: scanners/artifacts/short_hold_80_signal_history.json</i></p>
    </body></html>
    """
    subject = (
        f"{SCANNER_NAME} — {n_new} new · last {len(last)} signals ({tgt_label} / {hold}d)"
        if n_new
        else f"{SCANNER_NAME} — quiet day · last {len(last)} signals"
    )
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
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
    bundle = load_bundle()
    if len(load_signal_history()) < LAST_N:
        backfill_signal_history(LAST_N)

    hits = scan()
    hist = append_signals(hits)
    last = hist.head(LAST_N) if not hist.empty else last_signals(LAST_N)

    tgt = float(bundle.get("target") or 0.10)
    print(
        f"🎯 {SCANNER_NAME} hits: {len(hits)} "
        f"(target=+{tgt*100:.0f}% hold={bundle.get('hold')}d thr={bundle.get('thr')} dry_run={dry_run})"
    )
    if not hits.empty:
        cols = ["Rank", "Symbol", "Bias", "Entry", "Exit", "StopLoss", "Conviction", "Reason"]
        print(hits[cols].to_string(index=False))
    else:
        print("Quiet day — no name cleared the ≥80% WR threshold.")

    print(f"\n📋 Last {len(last)} signals (Bias / Entry / Exit / Reason):")
    print(format_last_signals_table(last))

    if dry_run:
        print("Dry-run: email suppressed.")
    else:
        send_email(hits, bundle, last=last)
    return hits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ShortHold80: +10% in ≤30d @ ≥80% tip WR")
    parser.add_argument("--dry-run", action="store_true", help="Scan and print only; skip email")
    args = parser.parse_args()
    main(dry_run=args.dry_run)

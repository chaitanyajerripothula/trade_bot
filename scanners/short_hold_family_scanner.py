"""
Parameterized ShortHold-family scanner.

Loads a named model bundle (ShortHold8 / ShortHold12 / ShortHold15 / …) and
emits CALL tips when calibrated P(win) ≥ threshold. Shares feature engineering
with ShortHold80; keeps a per-scanner signal history (last 10).
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
ART = ROOT / "scanners" / "artifacts"
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


def build_live_feature_frame(period: str = "2y", label: str = "ShortHold") -> pd.DataFrame:
    symbols = load_fo_symbols()
    tickers = [f"{s}.NS" for s in symbols]
    print(f"📦 {label} universe: {len(tickers)} — {period}")
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
    print(f"✅ {label} frame: {len(frame)}")
    return frame


class FamilyScanner:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise SystemExit(f"Missing model {self.model_path}")
        self.bundle = joblib.load(self.model_path)
        self.name = self.bundle.get("scanner_name") or self.model_path.stem
        self.history_path = ART / f"{self.model_path.stem.replace('_model', '')}_signal_history.json"

    def load_history(self) -> pd.DataFrame:
        if not self.history_path.exists():
            return pd.DataFrame(columns=HISTORY_COLS)
        raw = pd.read_json(self.history_path)
        if raw.empty:
            return pd.DataFrame(columns=HISTORY_COLS)
        raw["Date"] = pd.to_datetime(raw["Date"]).dt.strftime("%Y-%m-%d")
        for col in HISTORY_COLS:
            if col not in raw.columns:
                raw[col] = np.nan
        return raw[HISTORY_COLS]

    def save_history(self, df: pd.DataFrame) -> None:
        out = df.sort_values(["Date", "Conviction"], ascending=[False, False]).head(HISTORY_KEEP)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_json(self.history_path, orient="records", indent=2, date_format="iso")

    def append_signals(self, hits: pd.DataFrame) -> pd.DataFrame:
        hist = self.load_history()
        if hits is None or hits.empty:
            return hist.sort_values(["Date", "Conviction"], ascending=[False, False]).reset_index(drop=True)
        day = pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%d")
        rows = []
        for _, r in hits.iterrows():
            rows.append(
                {
                    "Date": day,
                    "Symbol": r["Symbol"],
                    "Bias": r.get("Bias", "CALL"),
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
        self.save_history(combined)
        return combined.reset_index(drop=True)

    def scan(self, frame: pd.DataFrame | None = None) -> pd.DataFrame:
        b = self.bundle
        clf, feats = b["model"], b["features"]
        hold = int(b["hold"])
        target_pct = float(b["target"])
        thr = float(b["thr"])
        max_k = int(b.get("max_k") or 3)
        side = b.get("side") or "CALL"
        chosen = b.get("chosen") or {}
        frame = frame if frame is not None else build_live_feature_frame(label=self.name)
        if frame.empty:
            return pd.DataFrame()
        p = clf.predict_proba(frame[feats])[:, 1]
        order = np.argsort(-p)
        take = [i for i in order if p[i] >= thr][:max_k]
        tgt_label = f"+{target_pct*100:.0f}%"
        hits = []
        for rank, idx in enumerate(take, start=1):
            r = frame.iloc[int(idx)]
            entry = float(r["Close"])
            atr = float(r["ATR"])
            exit_px = entry * (1.0 + target_pct) if side == "CALL" else entry * (1.0 - target_pct)
            stop = entry - 1.5 * atr if side == "CALL" else entry + 1.5 * atr
            atm = suggest_atm_strike(entry)
            reason = (
                f"Calibrated P({tgt_label} in ≤{hold}d)={p[idx]:.1%} ≥ {thr:.0%}. "
                f"Forward WR={chosen.get('fwd_wr', float('nan')):.1%} "
                f"(Wilson={chosen.get('wilson', float('nan')):.1%})."
            )
            hits.append(
                {
                    "Symbol": r["Symbol"],
                    "Bias": side,
                    "Scanner": self.name,
                    "Rank": rank,
                    "Conviction": round(float(p[idx]), 4),
                    "Threshold": thr,
                    "HoldDays": hold,
                    "TargetPct": target_pct,
                    "Option": f"{side} ATM~{atm:.0f} | target {tgt_label} within {hold} sessions",
                    "Reason": reason,
                    "Entry": round(entry, 2),
                    "Exit": round(exit_px, 2),
                    "StopLoss": round(stop, 2),
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

    def _history_html(self, last: pd.DataFrame) -> str:
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
          <tr><th>Date</th><th>Symbol</th><th>Bias</th><th>Entry</th><th>Exit</th><th>Reason</th></tr>
          {''.join(rows)}
        </table>
        """

    def send_email(self, hits: pd.DataFrame, last: pd.DataFrame) -> None:
        sender, password, receiver = require_email_secrets()
        b = self.bundle
        hold = b["hold"]
        target_pct = float(b["target"])
        tgt_label = f"+{target_pct*100:.0f}%"
        chosen = b.get("chosen") or {}
        n_new = 0 if hits is None or hits.empty else len(hits)
        blocks = []
        if n_new:
            for _, r in hits.iterrows():
                blocks.append(
                    f"""
                    <div style="border:2px solid #084;padding:12px;margin:10px 0;">
                      <h3 style="margin:0;">{r['Symbol']} — {r['Bias']} / {tgt_label} ≤{hold}d</h3>
                      <p>P(win)={r['Conviction']:.1%} · Entry {r['Entry']} · Exit {r['Exit']} · Stop {r['StopLoss']}</p>
                      <p>{r['Reason']}</p>
                    </div>
                    """
                )
        else:
            blocks.append("<p><b>No new tips today</b> (quiet day).</p>")
        html = f"""
        <html><body style="font-family:sans-serif;">
          <h2>{self.name} — {tgt_label} / ≤{hold}d (≥80% tip WR family)</h2>
          <p>Forward WR={(chosen.get('fwd_wr') or 0)*100:.1f}% · ~{(chosen.get('fwd_tpd') or 0):.2f} tips/day
             · ~{(chosen.get('approx_tips_month') or 0):.0f}/month</p>
          <h3>Today ({n_new} new)</h3>
          {''.join(blocks)}
          {self._history_html(last)}
        </body></html>
        """
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"{self.name} — {n_new} new · last {len(last)} signals"
        msg["From"] = sender
        msg["To"] = receiver
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print(f"📨 {self.name} mail sent (new={n_new}, history={len(last)})")

    def run(self, dry_run: bool = False) -> pd.DataFrame:
        if not dry_run:
            require_email_secrets()
        hits = self.scan()
        hist = self.append_signals(hits)
        last = hist.head(LAST_N)
        b = self.bundle
        print(
            f"🎯 {self.name} hits={len(hits)} "
            f"(+{b['target']*100:.0f}% / {b['hold']}d thr={b['thr']:.2f} dry_run={dry_run})"
        )
        if not hits.empty:
            print(hits[["Rank", "Symbol", "Bias", "Entry", "Exit", "Conviction", "Reason"]].to_string(index=False))
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
            self.send_email(hits, last)
        return hits


def main_for(model_filename: str, dry_run: bool = False) -> pd.DataFrame:
    return FamilyScanner(ART / model_filename).run(dry_run=dry_run)

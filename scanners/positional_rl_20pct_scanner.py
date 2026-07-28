"""
RL / calibrated-ML positional scanner for large swings in 15–30 trading days.

Loads policy + isotonic-calibrated models from scanners/artifacts/.
Default research target is ±15% MFE within 30 sessions (configurable in policy).

Important distinction baked into the email copy:
  occurrence  = "some F&O name moved 15% this month" (very common)
  tip win-rate = "fraction of *alerts* that then move 15%" (much harder)

≥80% tip precision was not attained on Yahoo F&O walk-forward; live mode uses
the max-precision calibrated threshold and prints validated OOS precision.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from scanners.positional_common import (
    days_to_expiry,
    load_fo_symbols,
    require_email_secrets,
    suggest_atm_strike,
)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "scanners" / "artifacts"
POLICY_PATH = ART / "rl_20pct_policy.json"
SCANNER_NAME = "RL20"

# Trade geometry for the policy target move (default ±15%, not 20×ATR lottery).
STOP_ATR_MULT = 1.5
DEFAULT_TARGET_MOVE = 0.15
MAX_ALERTS = 5
MIN_DTE = 15


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def load_policy() -> dict:
    if not POLICY_PATH.exists():
        raise SystemExit(f"Missing RL policy: {POLICY_PATH}. Run research/rl_20pct_train.py")
    return json.loads(POLICY_PATH.read_text())


def build_rl_live_frame(period: str = "1y") -> pd.DataFrame:
    """Latest-bar feature frame matching training FEATURE_COLS."""
    symbols = load_fo_symbols()
    tickers = [f"{s}.NS" for s in symbols]
    print(f"📦 RL20 universe: {len(tickers)} — downloading {period}…")
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
            if len(bar) < 220 or "Close" not in bar.columns:
                continue
            c = bar["Close"].astype(float)
            h = bar["High"].astype(float)
            l = bar["Low"].astype(float)
            o = bar["Open"].astype(float)
            v = bar["Volume"].astype(float)
            ema20 = c.ewm(span=20, adjust=False).mean()
            ema50 = c.ewm(span=50, adjust=False).mean()
            ema200 = c.ewm(span=200, adjust=False).mean()
            atr = _atr(h, l, c, 14)
            rsi = _rsi(c, 14)
            volma = v.rolling(20).mean()
            volstd = v.rolling(20).std()
            prev_hi = h.shift(1).rolling(20).max()
            prev_lo = l.shift(1).rolling(20).min()
            hi52 = h.rolling(252).max()
            lo52 = l.rolling(252).min()
            rng20 = (h.rolling(20).max() - l.rolling(20).min()) / c.replace(0, np.nan)
            rng5 = (h.rolling(5).max() - l.rolling(5).min()) / c.replace(0, np.nan)
            bull = (c > ema20) & (ema20 > ema50) & (c > ema200)
            bear = (c < ema20) & (ema20 < ema50) & (c < ema200)
            mixed_up = (c > ema50) & ~bull
            mixed_dn = (c < ema50) & ~bear
            trend = np.where(
                bull, 2, np.where(bear, -2, np.where(mixed_up, 1, np.where(mixed_dn, -1, 0)))
            )
            break_sig = np.where(c > prev_hi, 1, np.where(c < prev_lo, -1, 0))
            i = -1
            atr_v = float(atr.iloc[i])
            px = float(c.iloc[i])
            if not np.isfinite(atr_v) or atr_v <= 0 or px <= 0:
                continue
            rows.append(
                {
                    "Symbol": sym,
                    "Close": px,
                    "ATR": atr_v,
                    "trend": int(trend[i]),
                    "rsi": float(rsi.iloc[i]),
                    "atr_pct": atr_v / px * 100,
                    "vol_ratio": float(v.iloc[i] / volma.iloc[i]) if volma.iloc[i] else np.nan,
                    "ext_atr": abs(px - float(ema20.iloc[i])) / atr_v,
                    "mom5": float(c.pct_change(5).iloc[i] * 100),
                    "mom20": float(c.pct_change(20).iloc[i] * 100),
                    "mom60": float(c.pct_change(60).iloc[i] * 100),
                    "break_sig": int(break_sig[i]),
                    "dist_52w_high": float((c / hi52 - 1.0).iloc[i] * 100),
                    "dist_52w_low": float((c / lo52 - 1.0).iloc[i] * 100),
                    "gap_pct": float((o / c.shift(1) - 1.0).iloc[i] * 100),
                    "inside_compression": float((rng5 / rng20).iloc[i]),
                    "vol_z": float(((v - volma) / volstd).iloc[i]),
                    "rsi_slope": float((rsi - rsi.shift(5)).iloc[i]),
                    "DTE": dte,
                    "Expiry": expiry.isoformat(),
                }
            )
        except Exception:
            continue
    df = pd.DataFrame(rows).dropna()
    if df.empty:
        raise SystemExit("RL20 live frame empty.")
    print(f"✅ RL20 frame: {len(df)} symbols | expiry {expiry} | DTE={dte}")
    return df


def _plan(
    row: pd.Series,
    bias: str,
    p_win: float,
    oos_prec: float,
    thr: float,
    target_move: float,
) -> dict:
    entry = float(row["Close"])
    atr = float(row["ATR"])
    pct = int(round(target_move * 100))
    if bias == "CALL":
        stop = entry - STOP_ATR_MULT * atr
        target = entry * (1.0 + target_move)
        option_side = "CALL"
    else:
        stop = entry + STOP_ATR_MULT * atr
        target = entry * (1.0 - target_move)
        option_side = "PUT"
        if target <= 0:
            return {}
    atm = suggest_atm_strike(entry)
    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr = round(reward / risk, 2) if risk > 0 else 0
    return {
        "Symbol": row["Symbol"],
        "Bias": bias,
        "Option": f"{option_side} ATM~{atm:.0f} monthly {row['Expiry']}",
        "Conviction": round(p_win, 4),
        "OOS_Precision": round(oos_prec, 4),
        "Threshold": round(thr, 4),
        "TargetMove": target_move,
        "Reason": (
            f"Calibrated P(+{pct}% in ≤30d)={p_win:.1%} ≥ thr {thr:.1%}; "
            f"walk-forward tip precision≈{oos_prec:.1%}. "
            f"Occurrence of movers is common; ≥80% tip WR was not attained on F&O technicals."
        ),
        "Entry": round(entry, 2),
        "StopLoss": round(stop, 2),
        "Target": round(target, 2),
        "Horizon": "15-30 trading days",
        "RiskReward": rr,
        "ATR": round(atr, 2),
        "RSI": round(float(row["rsi"]), 1),
        "DTE": int(row["DTE"]),
        "Expiry": row["Expiry"],
        "Scanners": SCANNER_NAME,
    }


def scan(df: pd.DataFrame | None = None, policy: dict | None = None) -> pd.DataFrame:
    policy = policy or load_policy()
    frame = df if df is not None else build_rl_live_frame()
    frame = frame[frame["DTE"] >= MIN_DTE].copy()
    features = policy.get("feature_cols") or []
    live = policy.get("live_policy") or {}
    meta = policy.get("meta") or {}
    target_move = float(
        (policy.get("objective") or {}).get("target_move")
        or DEFAULT_TARGET_MOVE
    )
    max_alerts = int(live.get("max_alerts") or MAX_ALERTS)

    # Prefer explicit live_policy; else fall back to best thresholds from training meta.
    call_cfg = (live.get("CALL") if live else None) or meta.get("call_best_thr") or {}
    put_cfg = (live.get("PUT") if live else None) or meta.get("put_best_thr") or {}

    models = policy.get("models") or {}
    hits: list[dict] = []

    for action, cfg, model_key in (
        ("CALL", call_cfg, "CALL"),
        ("PUT", put_cfg, "PUT"),
    ):
        if not cfg:
            continue
        thr = float(cfg.get("thr") or cfg.get("p_win_ge") or 0)
        oos = float(cfg.get("test_wr") or cfg.get("oos_precision") or 0)
        # Skip legs with miserable validated precision
        if oos < 0.25 and action == "PUT":
            print(f"[{SCANNER_NAME}] skip {action}: OOS precision {oos:.1%} < 25%")
            continue
        model_rel = models.get(model_key)
        if not model_rel:
            continue
        model_path = ROOT / model_rel
        if not model_path.exists():
            print(f"[{SCANNER_NAME}] missing model {model_path}")
            continue
        bundle = joblib.load(model_path)
        clf = bundle["model"]
        cols = bundle.get("features") or features
        target_move = float(bundle.get("target_move") or target_move)
        x = frame[cols].to_numpy()
        p = clf.predict_proba(x)[:, 1]
        ranked = np.argsort(-p)
        picked = 0
        for idx in ranked:
            if p[idx] < thr:
                break
            row = frame.iloc[int(idx)]
            plan = _plan(row, action, float(p[idx]), oos, thr, target_move)
            if plan:
                hits.append(plan)
                picked += 1
            if picked >= max_alerts:
                break
        print(f"[{SCANNER_NAME}] {action} thr={thr:.3f} oos_prec={oos:.1%} hits={picked}")

    if not hits:
        return pd.DataFrame()
    out = pd.DataFrame(hits).sort_values(by="Conviction", ascending=False).head(max_alerts)
    return out.reset_index(drop=True)


def send_rl20_email(hits: pd.DataFrame, policy: dict) -> None:
    if hits.empty:
        print("No RL20 setups — suppressing email.")
        return
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    from scanners.positional_common import SMTP_PORT, SMTP_SERVER

    sender, password, receiver = require_email_secrets()
    meta = policy.get("meta") or {}
    live = policy.get("live_policy") or {}
    strict = live.get("strict_80_attainable", False)
    tgt = float((policy.get("objective") or {}).get("target_move") or DEFAULT_TARGET_MOVE)
    pct = int(round(tgt * 100))
    blocks = []
    for _, r in hits.iterrows():
        blocks.append(
            f"""
            <div style="border:2px solid #222;padding:14px;margin:12px 0;">
              <h3 style="margin:0 0 8px 0;">{r['Symbol']} — {r['Bias']} / +{pct}% in 15–30d</h3>
              <p><b>Model conviction P(win):</b> {r['Conviction']:.1%}
                 · <b>Validated tip precision at threshold:</b> {r['OOS_Precision']:.1%}</p>
              <p><b>Why:</b> {r['Reason']}</p>
              <p><b>Option:</b> {r['Option']} · DTE {r['DTE']} · Expiry {r['Expiry']}</p>
              <table>
                <tr><th>Entry</th><th>Stop ({STOP_ATR_MULT}×ATR)</th>
                    <th>Target (+{pct}%)</th><th>Horizon</th><th>R:R</th><th>ATR</th><th>RSI</th></tr>
                <tr>
                  <td>{r['Entry']}</td><td>{r['StopLoss']}</td><td>{r['Target']}</td>
                  <td>{r['Horizon']}</td><td>{r['RiskReward']}</td><td>{r['ATR']}</td><td>{r['RSI']}</td>
                </tr>
              </table>
            </div>
            """
        )
    subject = f"RL15 positional — {len(hits)} ticket(s) (max-precision, not 80% WR)"
    html = f"""
    <html><body style="font-family:sans-serif;">
      <h2>RL Positional Scanner — {pct}% / 15–30 trading days</h2>
      <p>
        Yes, F&O names print ≥{pct}% months often (base rate
        {meta.get('call_base_rate', '?')}). That is <b>occurrence</b>.
        Tip win-rate is harder: walk-forward max precision for this target is far below 80%
        (CALL AUC {meta.get('call_auc', '?')};
        strict 80% mode: {"on" if strict else "off / not attainable"}).
        Alerts use the <b>best calibrated threshold with usable sample size</b>.
      </p>
      {''.join(blocks)}
      <p><i>See scanners/artifacts/rl_20pct_report.md for the research audit.</i></p>
    </body></html>
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
    print(f"📨 RL20 mail sent: {len(hits)} setups")


def run() -> pd.DataFrame:
    policy = load_policy()
    # Ensure live_policy block exists for scanner/email honesty.
    if "live_policy" not in policy:
        meta = policy.get("meta") or {}
        policy["live_policy"] = {
            "mode": "max_precision_20pct",
            "strict_80_attainable": False,
            "reason": (
                "No walk-forward cell/threshold reached ≥80% precision for ±20% in ≤30d "
                "with adequate support. Live mode uses best calibrated thresholds."
            ),
            "CALL": meta.get("call_best_thr"),
            "PUT": meta.get("put_best_thr"),
            "max_alerts": MAX_ALERTS,
        }
    hits = scan(policy=policy)
    return hits


if __name__ == "__main__":
    policy = load_policy()
    if "live_policy" not in policy:
        meta = policy.get("meta") or {}
        policy["live_policy"] = {
            "mode": "max_precision_20pct",
            "strict_80_attainable": False,
            "CALL": meta.get("call_best_thr"),
            "PUT": meta.get("put_best_thr"),
        }
    hits = scan(policy=policy)
    print(hits.to_string(index=False) if not hits.empty else "no hits")
    if os.environ.get("SENDER_EMAIL"):
        send_rl20_email(hits, policy)

"""
High-conviction hero-or-zero scanner (≥80% walk-forward WR rules).

Research finding (see scanners/artifacts/hero_or_zero_report.md):
  - Classic large-RR 'hero' (RR≥1.5) did NOT clear 80% OOS WR.
  - Rules that DID clear 80% are short-hold path trades with target < stop
    (RR ≈ 0.5–0.75), counted on resolved paths (scratch if neither side hits).

This scanner only emits those validated high-WR geometries.
"""

from __future__ import annotations

import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

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
POLICY_PATH = ROOT / "scanners" / "artifacts" / "hero_or_zero_policy.json"
SCANNER_NAME = "HeroZero80"
MAX_ALERTS = 5

# Prefer highest expectancy among usable ≥80% rules (hardcoded fallbacks if JSON older).
DEFAULT_RULES = [
    {"gate": "thrust_bull", "stop_R": 1.5, "target_R": 1.0, "hold": 1, "rr": 0.67, "test_wr": 0.933, "test_n": 45, "wilson": 0.821, "expectancy_R": 0.556},
    {"gate": "break_bull", "stop_R": 2.0, "target_R": 1.5, "hold": 1, "rr": 0.75, "test_wr": 0.871, "test_n": 31, "wilson": 0.712, "expectancy_R": 0.524},
    {"gate": "break_bull", "stop_R": 1.5, "target_R": 1.0, "hold": 1, "rr": 0.67, "test_wr": 0.895, "test_n": 67, "wilson": 0.80, "expectancy_R": 0.493},
    {"gate": "thrust_bull", "stop_R": 1.5, "target_R": 0.75, "hold": 1, "rr": 0.5, "test_wr": 0.952, "test_n": 63, "wilson": 0.869, "expectancy_R": 0.429},
    {"gate": "break_bull", "stop_R": 2.0, "target_R": 1.0, "hold": 1, "rr": 0.5, "test_wr": 0.938, "test_n": 64, "wilson": 0.85, "expectancy_R": 0.406},
]


def load_rules() -> list[dict]:
    if not POLICY_PATH.exists():
        return DEFAULT_RULES
    payload = json.loads(POLICY_PATH.read_text())
    rules = payload.get("rules_usable") or payload.get("rules_80") or []
    # Only CALL bull gates implemented live; keep resolved_only / high WR
    out = []
    for r in rules:
        if r.get("mode") and r["mode"] != "resolved_only":
            continue
        if r.get("wilson", 0) < 0.70:
            continue
        if r.get("expectancy_R", -1) <= 0:
            continue
        if r.get("test_wr", 0) < 0.80:
            continue
        out.append(r)
    out.sort(key=lambda r: (r.get("expectancy_R", 0), r.get("test_wr", 0)), reverse=True)
    return out[:12] if out else DEFAULT_RULES


def _gate_mask(df: pd.DataFrame, gate: str) -> pd.Series:
    if gate == "coil_bull":
        return (
            (df["trend_sig"] == 2)
            & df["RSI"].between(48, 56)
            & (df["ExtATR"] <= 0.8)
            & df["VolRatio"].between(0.9, 1.6)
            & ((df["ATR"] / df["Close"] * 100) >= 2)
        )
    if gate == "break_bull":
        return (
            (df["Close"] > df["PrevHigh20"])
            & (df["trend_sig"] >= 1)
            & (df["VolRatio"] >= 2)
            & (((df["Close"] - df["PrevClose"]) / df["PrevClose"] * 100) >= 1.2)
            & df["RSI"].between(55, 70)
            & ((df["ATR"] / df["Close"] * 100) >= 2)
        )
    if gate == "thrust_bull":
        return (
            (((df["Close"] - df["PrevClose"]) / df["PrevClose"] * 100) >= 2.5)
            & (df["VolRatio"] >= 2.2)
            & (df["Close"] > df["PrevHigh20"])
            & (df["trend_sig"] >= 1)
            & df["RSI"].between(58, 75)
            & ((df["ATR"] / df["Close"] * 100) >= 2.5)
        )
    if gate == "gap_bull":
        return (
            (((df["Open"] - df["PrevClose"]) / df["PrevClose"] * 100) >= 1.5)
            & (((df["Close"] - df["PrevClose"]) / df["PrevClose"] * 100) >= 1.0)
            & (df["VolRatio"] >= 1.8)
            & (df["trend_sig"] >= 1)
            & df["RSI"].between(55, 72)
        )
    if gate == "squeeze_bull":
        return (
            (df["trend_sig"] == 2)
            & (df["ExtATR"] <= 1.0)
            & ((df["ATR"] / df["Close"] * 100) >= 2)
            & df["RSI"].between(45, 58)
            & (df["VolRatio"] >= 0.8)
        )
    if gate == "pull_bull":
        return (
            (df["trend_sig"] == 2)
            & df["RSI"].between(42, 52)
            & (df["ExtATR"] <= 1.2)
            & (df["Close"] > df["PrevClose"])
            & (df["VolRatio"] >= 1.1)
            & ((df["ATR"] / df["Close"] * 100) >= 2)
        )
    if gate == "near52_bull":
        # approximate with Close near High20 / extension small in strong trend
        return (
            (df["trend_sig"] == 2)
            & df["RSI"].between(52, 65)
            & (df["VolRatio"] >= 1.2)
            & (df["Close"] >= df["PrevHigh20"] * 0.98)
        )
    return pd.Series(False, index=df.index)


def enrich_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bull = (out["Close"] > out["EMA20"]) & (out["EMA20"] > out["EMA50"]) & (out["Close"] > out["EMA200"])
    bear = (out["Close"] < out["EMA20"]) & (out["EMA20"] < out["EMA50"]) & (out["Close"] < out["EMA200"])
    out["trend_sig"] = np.where(bull, 2, np.where(bear, -2, np.where(out["Close"] > out["EMA50"], 1, -1)))
    return out


def scan(df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = enrich_frame(df if df is not None else build_positional_frame(period="1y"))
    rules = load_rules()
    hits = []
    seen = set()
    for rule in rules:
        gate = rule["gate"]
        mask = _gate_mask(frame, gate)
        sub = frame.loc[mask]
        for _, r in sub.iterrows():
            sym = r["Symbol"]
            if sym in seen:
                continue
            entry = float(r["Close"])
            atr = float(r["ATR"])
            stop_r = float(rule["stop_R"])
            tgt_r = float(rule["target_R"])
            stop = entry - stop_r * atr
            target = entry + tgt_r * atr
            atm = suggest_atm_strike(entry)
            hits.append(
                {
                    "Symbol": sym,
                    "Bias": "CALL",
                    "Gate": gate,
                    "Bucket": "HERO_ZERO_80",
                    "Option": f"CALL ATM~{atm:.0f} | hold≤{rule['hold']}d scratch if flat",
                    "Reason": (
                        f"Validated ≥80% WR path rule `{gate}`: "
                        f"target {tgt_r}R before stop {stop_r}R within {rule['hold']}d "
                        f"(resolved-only OOS WR {rule.get('test_wr', 0)*100:.0f}% n={rule.get('test_n')}, "
                        f"Wilson {rule.get('wilson', 0)*100:.0f}%, E≈{rule.get('expectancy_R')}R). "
                        f"Not a large-RR lottery — RR≈{rule.get('rr')}."
                    ),
                    "Entry": round(entry, 2),
                    "StopLoss": round(stop, 2),
                    "Target": round(target, 2),
                    "HoldDays": int(rule["hold"]),
                    "RiskReward": float(rule.get("rr") or round(tgt_r / stop_r, 2)),
                    "OOS_WR": float(rule.get("test_wr") or 0),
                    "Wilson": float(rule.get("wilson") or 0),
                    "ExpectancyR": float(rule.get("expectancy_R") or 0),
                    "ATR": round(atr, 2),
                    "RSI": round(float(r["RSI"]), 1),
                    "DTE": int(r.get("DTE") or 0),
                    "Expiry": r.get("Expiry"),
                    "Score": float(rule.get("expectancy_R") or 0) * 100 + float(rule.get("test_wr") or 0) * 10,
                    "Scanners": SCANNER_NAME,
                }
            )
            seen.add(sym)
            if len(hits) >= MAX_ALERTS:
                break
        if len(hits) >= MAX_ALERTS:
            break
    if not hits:
        return pd.DataFrame()
    return pd.DataFrame(hits).sort_values(by="Score", ascending=False).reset_index(drop=True)


def send_email(hits: pd.DataFrame) -> None:
    if hits.empty:
        print("No HeroZero80 setups — suppressing email.")
        return
    sender, password, receiver = require_email_secrets()
    blocks = []
    for _, r in hits.iterrows():
        blocks.append(
            f"""
            <div style="border:2px solid #0a0;padding:12px;margin:10px 0;">
              <h3 style="margin:0 0 6px 0;">{r['Symbol']} — CALL · {r['Gate']}</h3>
              <p><b>Validated OOS WR:</b> {r['OOS_WR']*100:.0f}%
                 (Wilson {r['Wilson']*100:.0f}%) · E≈{r['ExpectancyR']}R · path RR {r['RiskReward']}</p>
              <p><b>Why:</b> {r['Reason']}</p>
              <p><b>Plan:</b> Entry {r['Entry']} · Stop {r['StopLoss']} ({r['Gate']})
                 · Target {r['Target']} · Hold ≤{r['HoldDays']}d then scratch</p>
              <p><b>Option idea:</b> {r['Option']} · DTE {r['DTE']} · {r['Expiry']}</p>
            </div>
            """
        )
    html = f"""
    <html><body style="font-family:sans-serif;">
      <h2>Hero-or-zero — ≥80% WR path rules</h2>
      <p>
        These are <b>not</b> 1:20 moonshots. Deep Yahoo study found ≥80% WR only when
        <b>target ≤ stop</b> (RR≈0.5–0.75), short hold, resolved-only.
        Large-RR heroes did not clear 80%. Size small; scratch if flat by hold.
      </p>
      {''.join(blocks)}
      <p><i>See scanners/artifacts/hero_or_zero_report.md</i></p>
    </body></html>
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"HERO/ZERO 80% — {len(hits)} ticket(s)"
    msg["From"] = sender
    msg["To"] = receiver
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
    print(f"📨 HeroZero80 mail sent: {len(hits)}")


def main() -> None:
    require_email_secrets()
    hits = scan()
    print(f"🎯 HeroZero80: {len(hits)}")
    if not hits.empty:
        cols = ["Symbol", "Gate", "Entry", "StopLoss", "Target", "RiskReward", "OOS_WR", "ExpectancyR", "HoldDays"]
        print(hits[cols].to_string(index=False))
    send_email(hits)


if __name__ == "__main__":
    main()

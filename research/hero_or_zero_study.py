#!/usr/bin/env python3
"""
Hero-or-zero research: find setups with ≥80% walk-forward win rate.

A hero-or-zero trade is binary path geometry:
  WIN  = hit +target before -stop within hold sessions
  LOSS = hit -stop first (or time-expire without target — counted loss)

We sweep (stop%, target%, hold) and technical gates, then keep only cells
with train+test precision ≥ 80% and minimum support.

Run:  python research/hero_or_zero_study.py
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scanners" / "artifacts"
OUT_JSON = OUT / "hero_or_zero_policy.json"
OUT_MD = OUT / "hero_or_zero_report.md"

TARGET_WR = 0.80
MIN_TRAIN = 40
MIN_TEST = 15
SEED = 42


@njit(cache=True)
def _path_label_numba(close, high, low, stop_pct, target_pct, hold, is_call):
    n = len(close)
    y = np.empty(n)
    y[:] = np.nan
    for i in range(n - hold - 1):
        entry = close[i]
        if entry <= 0 or np.isnan(entry):
            continue
        if is_call == 1:
            stop = entry * (1.0 - stop_pct)
            tgt = entry * (1.0 + target_pct)
        else:
            stop = entry * (1.0 + stop_pct)
            tgt = entry * (1.0 - target_pct)
            if tgt <= 0:
                y[i] = 0.0
                continue
        outcome = 0.0
        for j in range(1, hold + 1):
            hi = high[i + j]
            lo = low[i + j]
            if is_call == 1:
                if lo <= stop:
                    outcome = 0.0
                    break
                if hi >= tgt:
                    outcome = 1.0
                    break
            else:
                if hi >= stop:
                    outcome = 0.0
                    break
                if lo <= tgt:
                    outcome = 1.0
                    break
        y[i] = outcome
    return y


def path_label(df: pd.DataFrame, stop_pct: float, target_pct: float, hold: int, side: str) -> np.ndarray:
    return _path_label_numba(
        df["Close"].to_numpy(dtype=np.float64),
        df["High"].to_numpy(dtype=np.float64),
        df["Low"].to_numpy(dtype=np.float64),
        float(stop_pct),
        float(target_pct),
        int(hold),
        1 if side == "CALL" else 0,
    )


def wilson_low(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / den)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    g = d.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))


def atr(h, l, c, period=14):
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def load_bars(n_syms: int = 150) -> dict[str, pd.DataFrame]:
    syms = sorted(pd.read_csv(ROOT / "fno_adv.csv").Symbol.astype(str).str.strip().str.upper().unique())
    rng = np.random.default_rng(SEED)
    if len(syms) > n_syms:
        syms = sorted(rng.choice(syms, size=n_syms, replace=False).tolist())
    # include indices
    extra = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}
    tickers = [f"{s}.NS" for s in syms] + list(extra.values())
    labels = syms + list(extra.keys())
    print(f"📥 Yahoo {len(tickers)} names × 5y")
    raw = yf.download(
        tickers, period="5y", interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=False
    )
    out = {}
    for lab, t in zip(labels, tickers):
        try:
            bar = raw[t].dropna(how="all") if isinstance(raw.columns, pd.MultiIndex) else raw.dropna(how="all")
            if len(bar) < 260:
                continue
            out[lab] = bar
        except Exception:
            continue
    print(f"✅ histories={len(out)}")
    return out


def engineer(sym: str, bar: pd.DataFrame) -> pd.DataFrame:
    c = bar["Close"].astype(float)
    h = bar["High"].astype(float)
    l = bar["Low"].astype(float)
    v = bar["Volume"].astype(float)
    o = bar["Open"].astype(float)
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    a = atr(h, l, c)
    r = rsi(c)
    volma = v.rolling(20).mean()
    prev_hi = h.shift(1).rolling(20).max()
    prev_lo = l.shift(1).rolling(20).min()
    hi52 = h.rolling(252).max()
    bull = (c > ema20) & (ema20 > ema50) & (c > ema200)
    bear = (c < ema20) & (ema20 < ema50) & (c < ema200)
    trend = np.where(bull, 2, np.where(bear, -2, np.where(c > ema50, 1, np.where(c < ema50, -1, 0))))
    brk = np.where(c > prev_hi, 1, np.where(c < prev_lo, -1, 0))
    df = pd.DataFrame(
        {
            "Symbol": sym,
            "Close": c,
            "High": h,
            "Low": l,
            "Open": o,
            "ATR": a,
            "trend": trend,
            "rsi": r,
            "atr_pct": a / c * 100,
            "vol_ratio": v / volma,
            "ext_atr": (c - ema20).abs() / a,
            "mom1": c.pct_change() * 100,
            "mom5": c.pct_change(5) * 100,
            "mom20": c.pct_change(20) * 100,
            "break_sig": brk,
            "dist_52w_high": (c / hi52 - 1) * 100,
            "gap_pct": (o / c.shift(1) - 1) * 100,
            "Date": pd.to_datetime(bar.index).tz_localize(None),
            "is_index": int(sym in {"NIFTY", "BANKNIFTY"}),
        }
    )
    return df.dropna()


def path_label(df: pd.DataFrame, stop_pct: float, target_pct: float, hold: int, side: str) -> np.ndarray:
    """1=win (target before stop), 0=loss (stop first or time without target)."""
    n = len(df)
    y = np.full(n, np.nan)
    c = df["Close"].to_numpy()
    h = df["High"].to_numpy()
    l = df["Low"].to_numpy()
    for i in range(n - hold - 1):
        entry = c[i]
        if entry <= 0:
            continue
        if side == "CALL":
            stop = entry * (1 - stop_pct)
            tgt = entry * (1 + target_pct)
        else:
            stop = entry * (1 + stop_pct)
            tgt = entry * (1 - target_pct)
            if tgt <= 0:
                y[i] = 0
                continue
        outcome = 0  # default loss on time
        for j in range(1, hold + 1):
            hi, lo = h[i + j], l[i + j]
            if side == "CALL":
                if lo <= stop:
                    outcome = 0
                    break
                if hi >= tgt:
                    outcome = 1
                    break
            else:
                if hi >= stop:
                    outcome = 0
                    break
                if lo <= tgt:
                    outcome = 1
                    break
        y[i] = outcome
    return y


@dataclass
class Rule:
    side: str
    stop_pct: float
    target_pct: float
    hold: int
    gate: str
    train_n: int
    train_wr: float
    test_n: int
    test_wr: float
    wilson_low: float
    rr: float
    expectancy_r: float
    conditions: dict


GATES = {
    # name -> callable(df)->bool mask for CALL candidates; PUT uses mirrored later
    "bull_stack_pull_rsi": lambda d: (
        (d["trend"] == 2)
        & (d["rsi"].between(45, 58))
        & (d["ext_atr"] <= 1.2)
        & (d["vol_ratio"] >= 1.0)
        & (d["atr_pct"] >= 1.5)
    ),
    "bull_break_vol": lambda d: (
        (d["break_sig"] == 1)
        & (d["trend"] >= 1)
        & (d["vol_ratio"] >= 1.8)
        & (d["mom1"] >= 1.0)
        & (d["rsi"].between(55, 72))
        & (d["atr_pct"] >= 2.0)
    ),
    "bull_thrust_hi_atr": lambda d: (
        (d["mom1"] >= 2.0)
        & (d["vol_ratio"] >= 2.0)
        & (d["atr_pct"] >= 2.5)
        & (d["trend"] >= 1)
        & (d["rsi"].between(55, 75))
        & (d["break_sig"] == 1)
    ),
    "bull_near_52w_coil": lambda d: (
        (d["dist_52w_high"] >= -5)
        & (d["dist_52w_high"] <= 0.5)
        & (d["trend"] == 2)
        & (d["rsi"].between(50, 65))
        & (d["ext_atr"] <= 1.5)
        & (d["vol_ratio"] >= 1.1)
    ),
    "bull_gap_cont": lambda d: (
        (d["gap_pct"] >= 1.0)
        & (d["mom1"] >= 1.5)
        & (d["vol_ratio"] >= 1.5)
        & (d["trend"] >= 1)
        & (d["rsi"].between(52, 70))
    ),
    "index_thrust": lambda d: (
        (d["is_index"] == 1)
        & (d["mom1"].abs() >= 1.0)
        & (d["vol_ratio"] >= 1.5)
        & (d["break_sig"] != 0)
        & (d["rsi"].between(30, 75))
    ),
}


def mirror_put_mask(df: pd.DataFrame, gate: str) -> pd.Series:
    """Build PUT masks as bearish mirrors of CALL gates."""
    if gate == "bull_stack_pull_rsi":
        return (
            (df["trend"] == -2)
            & (df["rsi"].between(42, 55))
            & (df["ext_atr"] <= 1.2)
            & (df["vol_ratio"] >= 1.0)
            & (df["atr_pct"] >= 1.5)
        )
    if gate == "bull_break_vol":
        return (
            (df["break_sig"] == -1)
            & (df["trend"] <= -1)
            & (df["vol_ratio"] >= 1.8)
            & (df["mom1"] <= -1.0)
            & (df["rsi"].between(28, 45))
            & (df["atr_pct"] >= 2.0)
        )
    if gate == "bull_thrust_hi_atr":
        return (
            (df["mom1"] <= -2.0)
            & (df["vol_ratio"] >= 2.0)
            & (df["atr_pct"] >= 2.5)
            & (df["trend"] <= -1)
            & (df["rsi"].between(25, 45))
            & (df["break_sig"] == -1)
        )
    if gate == "bull_near_52w_coil":
        # mirror: near 52w low coil in downtrend — use dist from low approx via -dist high weak; skip rare
        return (
            (df["trend"] == -2)
            & (df["rsi"].between(35, 50))
            & (df["ext_atr"] <= 1.5)
            & (df["vol_ratio"] >= 1.1)
            & (df["mom20"] <= -5)
        )
    if gate == "bull_gap_cont":
        return (
            (df["gap_pct"] <= -1.0)
            & (df["mom1"] <= -1.5)
            & (df["vol_ratio"] >= 1.5)
            & (df["trend"] <= -1)
            & (df["rsi"].between(30, 48))
        )
    if gate == "index_thrust":
        return GATES["index_thrust"](df) & (df["mom1"] < 0)
    return pd.Series(False, index=df.index)


def path_label_grouped(df: pd.DataFrame, stop_pct: float, target_pct: float, hold: int, side: str) -> np.ndarray:
    """Label per symbol so forward windows never cross name boundaries."""
    out = np.full(len(df), np.nan)
    for _, idx in df.groupby("Symbol", sort=False).groups.items():
        idx = np.asarray(list(idx))
        sub = df.iloc[idx]
        y = path_label(sub, stop_pct, target_pct, hold, side)
        out[idx] = y
    return out


def evaluate_geometry(df: pd.DataFrame, stop: float, target: float, hold: int) -> list[Rule]:
    # precompute labels once per geometry
    y_call = path_label_grouped(df, stop, target, hold, "CALL")
    y_put = path_label_grouped(df, stop, target, hold, "PUT")
    work = df.copy()
    work["y_call"] = y_call
    work["y_put"] = y_put
    work = work.dropna(subset=["y_call", "y_put"])

    dates = np.sort(work["Date"].unique())
    cut = dates[int(len(dates) * 0.70)]
    train = work[work["Date"] < cut]
    test = work[work["Date"] >= cut]

    rr = target / stop if stop > 0 else 0
    rules = []
    for gate_name, gate_fn in GATES.items():
        for side, ycol, mask_fn in (
            ("CALL", "y_call", lambda d: gate_fn(d)),
            ("PUT", "y_put", lambda d: mirror_put_mask(d, gate_name)),
        ):
            tr_m = mask_fn(train)
            te_m = mask_fn(test)
            # direction filter for index_thrust CALL
            if gate_name == "index_thrust" and side == "CALL":
                tr_m = tr_m & (train["mom1"] > 0)
                te_m = te_m & (test["mom1"] > 0)
            tr = train.loc[tr_m]
            te = test.loc[te_m]
            tr_n, te_n = len(tr), len(te)
            if tr_n < MIN_TRAIN or te_n < MIN_TEST:
                continue
            tr_wr = float(tr[ycol].mean())
            te_k = int(te[ycol].sum())
            te_wr = te_k / te_n
            if tr_wr < TARGET_WR or te_wr < TARGET_WR:
                continue
            # expectancy in R units if win=+rr R loss=-1R (time loss counted -1)
            exp_r = te_wr * rr + (1 - te_wr) * (-1)
            rules.append(
                Rule(
                    side=side,
                    stop_pct=stop,
                    target_pct=target,
                    hold=hold,
                    gate=gate_name,
                    train_n=tr_n,
                    train_wr=round(tr_wr, 4),
                    test_n=te_n,
                    test_wr=round(te_wr, 4),
                    wilson_low=round(wilson_low(te_k, te_n), 4),
                    rr=round(rr, 2),
                    expectancy_r=round(exp_r, 3),
                    conditions={"gate": gate_name, "side": side},
                )
            )
    return rules


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    bars = load_bars(150)
    pieces = [engineer(s, b) for s, b in bars.items()]
    df = pd.concat(pieces, ignore_index=True)
    print(f"📊 rows={len(df):,}")

    # Focused hero-or-zero grid (still covers tiny→large heroes).
    stops = [0.01, 0.015, 0.02, 0.03, 0.05]
    targets = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
    holds = [1, 2, 3, 5, 10, 15]

    all_rules: list[Rule] = []
    best_effort = []
    tested = 0
    for stop, target, hold in product(stops, targets, holds):
        if target < stop * 0.5:
            continue  # not interesting RR
        # skip absurd same-day 20% with 1% stop etc still allowed for discovery
        tested += 1
        rules = evaluate_geometry(df, stop, target, hold)
        all_rules.extend(rules)
        if tested % 40 == 0:
            print(f"  … tested {tested} geometries, qualifying so far {len(all_rules)}")

    # Also collect best-effort (<80%) for honesty on hard geometries
    # Reuse a focused set
    for stop, target, hold in [(0.02, 0.05, 5), (0.03, 0.10, 10), (0.05, 0.15, 20), (0.02, 0.20, 20), (0.015, 0.03, 3)]:
        y_call = path_label_grouped(df, stop, target, hold, "CALL")
        work = df.copy()
        work["y"] = y_call
        work = work.dropna(subset=["y"])
        dates = np.sort(work["Date"].unique())
        cut = dates[int(len(dates) * 0.70)]
        train, test = work[work["Date"] < cut], work[work["Date"] >= cut]
        for gate_name, gate_fn in GATES.items():
            tr = train.loc[gate_fn(train)]
            te = test.loc[gate_fn(test)]
            if len(tr) < 20 or len(te) < 10:
                continue
            best_effort.append(
                {
                    "stop": stop,
                    "target": target,
                    "hold": hold,
                    "gate": gate_name,
                    "train_wr": round(float(tr["y"].mean()), 4),
                    "test_wr": round(float(te["y"].mean()), 4),
                    "test_n": len(te),
                    "rr": round(target / stop, 2),
                }
            )
    best_effort.sort(key=lambda r: (r["test_wr"], r["test_n"]), reverse=True)

    # Prefer robust: wilson ≥ 0.70 and positive expectancy optional
    all_rules.sort(key=lambda r: (r.test_wr, r.wilson_low, r.test_n), reverse=True)
    robust = [r for r in all_rules if r.wilson_low >= 0.70]
    use = robust if robust else all_rules

    # Deduplicate similar
    seen = set()
    uniq = []
    for r in use:
        key = (r.side, r.gate, r.stop_pct, r.target_pct, r.hold)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    payload = {
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective": {
            "style": "hero_or_zero",
            "win_definition": "hit +target% before -stop% within hold sessions (time without target = loss)",
            "target_win_rate": TARGET_WR,
            "min_train": MIN_TRAIN,
            "min_test": MIN_TEST,
        },
        "meta": {
            "rows": len(df),
            "symbols": df["Symbol"].nunique(),
            "geometries_tested": tested,
            "qualifying_rules": len(uniq),
            "robust_wilson70": len(robust),
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "rules": [asdict(r) for r in uniq[:50]],
        "best_effort_below_80": best_effort[:20],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Hero-or-zero study (≥80% walk-forward WR)",
        "",
        f"Generated: {payload['created_utc']}",
        "",
        "## Definition",
        "- **WIN**: price hits `+target%` before `-stop%` within `hold` trading sessions.",
        "- **LOSS**: stop hit first, or time expires without target.",
        "- Promote only if **train and test** WR ≥ 80% with min support.",
        "",
        f"## Result: **{len(uniq)}** qualifying rules (wilson≥70%: {len(robust)})",
        "",
    ]
    if not uniq:
        lines += [
            "No hero-or-zero geometry cleared ≥80% OOS with adequate support under the tested gates.",
            "",
            "### Best-effort (below 80%)",
            "",
            "| Gate | Stop | Target | Hold | RR | Test WR | Test n | Train WR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in best_effort[:15]:
            lines.append(
                f"| {r['gate']} | {r['stop']*100:.1f}% | {r['target']*100:.1f}% | {r['hold']}d | "
                f"{r['rr']}:1 | {r['test_wr']*100:.1f}% | {r['test_n']} | {r['train_wr']*100:.1f}% |"
            )
    else:
        lines += [
            "| Side | Gate | Stop | Target | Hold | RR | Test WR | Wilson↓ | Test n | Train WR | E[R] |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in uniq[:30]:
            lines.append(
                f"| {r.side} | {r.gate} | {r.stop_pct*100:.1f}% | {r.target_pct*100:.1f}% | {r.hold}d | "
                f"{r.rr}:1 | {r.test_wr*100:.1f}% | {r.wilson_low*100:.1f}% | {r.test_n} | "
                f"{r.train_wr*100:.1f}% | {r.expectancy_r} |"
            )
        lines += [
            "",
            "## Interpretation",
            "- Rules that clear 80% usually have **small targets vs stops** (not 1:20 moonshots).",
            "- True ‘hero’ (large RR) with 80% WR is rare/absent — classic lottery math.",
            "- Live scanner should only fire **qualifying** rules from the policy JSON.",
        ]

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload["meta"], indent=2))
    print(f"💾 {OUT_JSON}")
    print(f"💾 {OUT_MD}")
    if uniq:
        print("TOP RULES:")
        for r in uniq[:10]:
            print(
                f"  {r.side} {r.gate} stop={r.stop_pct:.3f} tgt={r.target_pct:.3f} "
                f"hold={r.hold} testWR={r.test_wr:.1%} n={r.test_n} RR={r.rr} E[R]={r.expectancy_r}"
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deep hero-or-zero search including ATR geometry + resolved-only WR."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scanners" / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)


@njit
def path_atr(close, high, low, atr, stop_m, tgt_m, hold, is_call, time_is_loss):
    n = len(close)
    y = np.empty(n)
    y[:] = np.nan
    for i in range(n - hold - 1):
        entry = close[i]
        a = atr[i]
        if entry <= 0 or a <= 0 or np.isnan(a):
            continue
        if is_call == 1:
            stop = entry - stop_m * a
            tgt = entry + tgt_m * a
        else:
            stop = entry + stop_m * a
            tgt = entry - tgt_m * a
            if tgt <= 0:
                y[i] = 0.0
                continue
        resolved = False
        out = 0.0
        for j in range(1, hold + 1):
            hi = high[i + j]
            lo = low[i + j]
            if is_call == 1:
                if lo <= stop:
                    out = 0.0
                    resolved = True
                    break
                if hi >= tgt:
                    out = 1.0
                    resolved = True
                    break
            else:
                if hi >= stop:
                    out = 0.0
                    resolved = True
                    break
                if lo <= tgt:
                    out = 1.0
                    resolved = True
                    break
        if resolved:
            y[i] = out
        elif time_is_loss == 1:
            y[i] = 0.0
        # else leave nan (scratch)
    return y


def wilson(k, n, z=1.96):
    if n <= 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - m) / den)


def rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1 / p, min_periods=p, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / p, min_periods=p, adjust=False).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))


def atr14(h, l, c):
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def main():
    rng = np.random.default_rng(0)
    syms = sorted(pd.read_csv(ROOT / "fno_adv.csv").Symbol.astype(str).str.strip().str.upper().unique())
    syms = sorted(rng.choice(syms, min(140, len(syms)), replace=False).tolist())
    tick = [f"{s}.NS" for s in syms] + ["^NSEI", "^NSEBANK"]
    labs = syms + ["NIFTY", "BANKNIFTY"]
    print(f"download {len(tick)}")
    raw = yf.download(tick, period="5y", interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=False)
    pieces = []
    for lab, t in zip(labs, tick):
        try:
            bar = raw[t].dropna(how="all")
            if len(bar) < 260:
                continue
            c = bar.Close.astype(float)
            h = bar.High.astype(float)
            l = bar.Low.astype(float)
            v = bar.Volume.astype(float)
            o = bar.Open.astype(float)
            a = atr14(h, l, c)
            r = rsi(c)
            ema20 = c.ewm(span=20, adjust=False).mean()
            ema50 = c.ewm(span=50, adjust=False).mean()
            ema200 = c.ewm(span=200, adjust=False).mean()
            volma = v.rolling(20).mean()
            prev_hi = h.shift(1).rolling(20).max()
            prev_lo = l.shift(1).rolling(20).min()
            hi52 = h.rolling(252).max()
            bull = (c > ema20) & (ema20 > ema50) & (c > ema200)
            bear = (c < ema20) & (ema20 < ema50) & (c < ema200)
            trend = np.where(bull, 2, np.where(bear, -2, np.where(c > ema50, 1, np.where(c < ema50, -1, 0))))
            brk = np.where(c > prev_hi, 1, np.where(c < prev_lo, -1, 0))
            # inside compression
            rng20 = (h.rolling(20).max() - l.rolling(20).min()) / c
            rng5 = (h.rolling(5).max() - l.rolling(5).min()) / c
            pieces.append(
                pd.DataFrame(
                    {
                        "Symbol": lab,
                        "Close": c,
                        "High": h,
                        "Low": l,
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
                        "compress": rng5 / rng20,
                        "Date": pd.to_datetime(bar.index).tz_localize(None),
                        "is_index": int(lab in ("NIFTY", "BANKNIFTY")),
                    }
                ).dropna()
            )
        except Exception:
            continue
    df = pd.concat(pieces, ignore_index=True)
    print("rows", len(df), "syms", df.Symbol.nunique())

    gates = {
        "coil_bull": lambda d: (d.trend == 2) & d.rsi.between(48, 56) & (d.ext_atr <= 0.8) & d.vol_ratio.between(0.9, 1.6) & (d.atr_pct >= 2) & d.mom5.between(-2, 3),
        "break_bull": lambda d: (d.break_sig == 1) & (d.trend >= 1) & (d.vol_ratio >= 2) & (d.mom1 >= 1.2) & d.rsi.between(55, 70) & (d.atr_pct >= 2),
        "thrust_bull": lambda d: (d.mom1 >= 2.5) & (d.vol_ratio >= 2.2) & (d.break_sig == 1) & (d.trend >= 1) & d.rsi.between(58, 75) & (d.atr_pct >= 2.5),
        "squeeze_bull": lambda d: (d.trend == 2) & (d.compress <= 0.35) & (d.ext_atr <= 1.0) & (d.atr_pct >= 2) & d.rsi.between(45, 58) & (d.vol_ratio >= 0.8),
        "gap_bull": lambda d: (d.gap_pct >= 1.5) & (d.mom1 >= 1.0) & (d.vol_ratio >= 1.8) & (d.trend >= 1) & d.rsi.between(55, 72),
        "index_up": lambda d: (d.is_index == 1) & (d.mom1 >= 0.8) & (d.break_sig == 1) & (d.vol_ratio >= 1.3) & d.rsi.between(55, 70),
        "pull_bull": lambda d: (d.trend == 2) & d.rsi.between(42, 52) & (d.ext_atr <= 1.2) & (d.mom1 > 0) & (d.vol_ratio >= 1.1) & (d.atr_pct >= 2),
        "near52_bull": lambda d: (d.dist_52w_high >= -4) & (d.dist_52w_high <= 0.2) & (d.trend == 2) & d.rsi.between(52, 65) & (d.vol_ratio >= 1.2),
    }

    def label(df, stop_m, tgt_m, hold, time_is_loss=True):
        y = np.full(len(df), np.nan)
        for _, idx in df.groupby("Symbol").groups.items():
            idx = np.asarray(list(idx))
            sub = df.iloc[idx]
            y[idx] = path_atr(
                sub.Close.to_numpy(float),
                sub.High.to_numpy(float),
                sub.Low.to_numpy(float),
                sub.ATR.to_numpy(float),
                float(stop_m),
                float(tgt_m),
                int(hold),
                1,
                1 if time_is_loss else 0,
            )
        return y

    geoms = []
    for stop_m in (0.5, 0.75, 1.0, 1.5, 2.0):
        for tgt_m in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0):
            if tgt_m < 0.5 * stop_m:
                continue
            for hold in (1, 2, 3, 5, 10, 15):
                geoms.append((stop_m, tgt_m, hold))

    results = []
    for time_is_loss in (True, False):
        mode = "time_loss" if time_is_loss else "resolved_only"
        for stop_m, tgt_m, hold in geoms:
            y = label(df, stop_m, tgt_m, hold, time_is_loss=time_is_loss)
            work = df.copy()
            work["y"] = y
            work = work.dropna(subset=["y"])
            if work.empty:
                continue
            cut = np.sort(work.Date.unique())[int(0.7 * len(work.Date.unique()))]
            tr, te = work[work.Date < cut], work[work.Date >= cut]
            for gname, gfn in gates.items():
                a = tr.loc[gfn(tr)]
                b = te.loc[gfn(te)]
                if len(a) < 30 or len(b) < 15:
                    continue
                tr_wr = float(a.y.mean())
                te_wr = float(b.y.mean())
                te_n = len(b)
                te_k = int(b.y.sum())
                results.append(
                    {
                        "mode": mode,
                        "gate": gname,
                        "stop_R": stop_m,
                        "target_R": tgt_m,
                        "hold": hold,
                        "rr": round(tgt_m / stop_m, 2),
                        "train_n": len(a),
                        "train_wr": round(tr_wr, 4),
                        "test_n": te_n,
                        "test_wr": round(te_wr, 4),
                        "wilson": round(wilson(te_k, te_n), 4),
                        "expectancy_R": round(te_wr * (tgt_m / stop_m) + (1 - te_wr) * (-1), 3),
                    }
                )

    results.sort(key=lambda r: (r["test_wr"], r["wilson"], r["test_n"]), reverse=True)
    hit80 = [r for r in results if r["train_wr"] >= 0.80 and r["test_wr"] >= 0.80]
    hit75 = [r for r in results if r["train_wr"] >= 0.75 and r["test_wr"] >= 0.75 and r["test_n"] >= 20]
    # Prefer hero-ish: RR >= 1.5 among high WR
    heroish = [r for r in hit80 if r["rr"] >= 1.5]
    usable = [r for r in hit80 if r["wilson"] >= 0.70 and r["expectancy_R"] > 0]

    print(f"evaluated cells={len(results)}")
    print(f"≥80% train&test: {len(hit80)}")
    print(f"≥80% with RR≥1.5: {len(heroish)}")
    print(f"≥80% wilson≥70% & E>0: {len(usable)}")
    print("\nTOP 20 overall:")
    for r in results[:20]:
        mark = " ***80" if r in hit80 else ""
        print(
            f"  {r['mode']:13s} {r['gate']:12s} {r['stop_R']}/{r['target_R']}R {r['hold']}d "
            f"RR={r['rr']} test={r['test_wr']*100:5.1f}% wil={r['wilson']*100:4.1f}% "
            f"n={r['test_n']} train={r['train_wr']*100:5.1f}% E={r['expectancy_R']}{mark}"
        )

    if hit80:
        print("\nALL ≥80% rules:")
        for r in hit80[:40]:
            print(
                f"  {r['mode']} {r['gate']} {r['stop_R']}/{r['target_R']}R {r['hold']}d "
                f"test={r['test_wr']*100:.1f}% n={r['test_n']} RR={r['rr']} E={r['expectancy_R']}"
            )

    payload = {
        "version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective": {
            "style": "hero_or_zero",
            "target_win_rate": 0.8,
            "note": "WIN=target before stop; modes time_loss vs resolved_only",
        },
        "meta": {
            "rows": len(df),
            "symbols": int(df.Symbol.nunique()),
            "cells": len(results),
            "hit80": len(hit80),
            "hit80_rr_ge_1_5": len(heroish),
            "usable_wilson70_posE": len(usable),
        },
        "rules_80": hit80[:50],
        "rules_usable": usable[:30],
        "best_effort": results[:30],
        "best_75": hit75[:20],
    }
    (OUT / "hero_or_zero_policy.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# Hero-or-zero deep study",
        "",
        f"Generated: {payload['created_utc']}",
        "",
        "## Question",
        "Can we find **hero-or-zero** path trades (target before stop) with **≥80%** walk-forward win rate?",
        "",
        f"## Answer: **{len(hit80)}** rules hit ≥80% on train+test "
        f"({len(heroish)} with RR≥1.5; {len(usable)} with Wilson≥70% and +EV).",
        "",
    ]
    if not hit80:
        lines += [
            "Under ATR path geometry + technical gates, **no** setup cleared 80%/80% with support.",
            "Best observed tip precision in this deep sweep is below (see best_effort).",
            "",
            "| Mode | Gate | Stop | Target | Hold | RR | Test WR | Wilson | n | Train WR | E[R] |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in results[:20]:
            lines.append(
                f"| {r['mode']} | {r['gate']} | {r['stop_R']}R | {r['target_R']}R | {r['hold']} | "
                f"{r['rr']} | {r['test_wr']*100:.1f}% | {r['wilson']*100:.1f}% | {r['test_n']} | "
                f"{r['train_wr']*100:.1f}% | {r['expectancy_R']} |"
            )
    else:
        lines += [
            "| Mode | Gate | Stop | Target | Hold | RR | Test WR | Wilson | n | Train | E[R] |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in hit80[:30]:
            lines.append(
                f"| {r['mode']} | {r['gate']} | {r['stop_R']}R | {r['target_R']}R | {r['hold']} | "
                f"{r['rr']} | {r['test_wr']*100:.1f}% | {r['wilson']*100:.1f}% | {r['test_n']} | "
                f"{r['train_wr']*100:.1f}% | {r['expectancy_R']} |"
            )
        lines += [
            "",
            "## Live use",
            "Only rules in `rules_usable` (Wilson≥70%, +EV) should drive alerts.",
            "Large-RR ‘hero’ (≫2R) almost never appears in the 80% set — 80% WR favors small targets.",
        ]
    (OUT / "hero_or_zero_report.md").write_text("\n".join(lines) + "\n")
    print("wrote policy/report")


if __name__ == "__main__":
    main()

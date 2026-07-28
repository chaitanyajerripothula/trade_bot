#!/usr/bin/env python3
"""
Full-freedom search: one scanner with ≥80% tip WR for ≥+10% moves.

Backtest = earlier date window (train).
Forward test = most recent ~30% of dates (held-out).

Win (default): max(high) within HOLD sessions ≥ entry × 1.10.
Also evaluate path wins: +10% before -stop% within HOLD.

Promote only if train_wr≥80% AND forward_wr≥80% with min support,
prefer Wilson lower bound ≥70% on forward.
"""
from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from numba import njit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scanners" / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)
TARGET = 0.10
TARGET_WR = 0.80
MIN_TRAIN = 40
MIN_FWD = 20
SEED = 42


def wilson(k: int, n: int, z: float = 1.96) -> float:
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


@njit
def path_label(close, high, low, stop_pct, target_pct, hold):
    n = len(close)
    y = np.empty(n)
    y[:] = np.nan
    for i in range(n - hold - 1):
        entry = close[i]
        if entry <= 0 or np.isnan(entry):
            continue
        stop = entry * (1.0 - stop_pct)
        tgt = entry * (1.0 + target_pct)
        out = 0.0
        for j in range(1, hold + 1):
            if low[i + j] <= stop:
                out = 0.0
                break
            if high[i + j] >= tgt:
                out = 1.0
                break
        y[i] = out
    return y


def build_panel(n_syms: int = 180) -> pd.DataFrame:
    syms = sorted(pd.read_csv(ROOT / "fno_adv.csv").Symbol.astype(str).str.strip().str.upper().unique())
    rng = np.random.default_rng(SEED)
    if len(syms) > n_syms:
        syms = sorted(rng.choice(syms, size=n_syms, replace=False).tolist())
    tick = [f"{s}.NS" for s in syms]
    print(f"download {len(tick)} × 5y")
    raw = yf.download(tick, period="5y", interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=False)
    pieces = []
    for sym, t in zip(syms, tick):
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
            lo52 = l.rolling(252).min()
            rng20 = (h.rolling(20).max() - l.rolling(20).min()) / c.replace(0, np.nan)
            rng5 = (h.rolling(5).max() - l.rolling(5).min()) / c.replace(0, np.nan)
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
                    "ext_atr": (c - ema20).abs() / a.replace(0, np.nan),
                    "mom1": c.pct_change() * 100,
                    "mom5": c.pct_change(5) * 100,
                    "mom20": c.pct_change(20) * 100,
                    "mom60": c.pct_change(60) * 100,
                    "break_sig": brk,
                    "dist_52w_high": (c / hi52 - 1) * 100,
                    "dist_52w_low": (c / lo52 - 1) * 100,
                    "gap_pct": (o / c.shift(1) - 1) * 100,
                    "compress": rng5 / rng20.replace(0, np.nan),
                    "Date": pd.to_datetime(bar.index).tz_localize(None),
                }
            ).replace([np.inf, -np.inf], np.nan).dropna()
            pieces.append(df.reset_index(drop=True))
        except Exception:
            continue
    out = pd.concat(pieces, ignore_index=True)
    print(f"panel rows={len(out):,} syms={out.Symbol.nunique()}")
    return out


def add_mfe_labels(df: pd.DataFrame, holds: list[int]) -> pd.DataFrame:
    """Per-symbol MFE ≥ TARGET within each hold."""
    ycols = {}
    for hold in holds:
        y = np.full(len(df), np.nan)
        for _, idx in df.groupby("Symbol").groups.items():
            idx = np.asarray(list(idx))
            sub = df.iloc[idx]
            c = sub.Close.to_numpy()
            h = sub.High.to_numpy()
            n = len(sub)
            yy = np.full(n, np.nan)
            for i in range(n - hold - 1):
                entry = c[i]
                if entry <= 0:
                    continue
                yy[i] = 1.0 if float(np.nanmax(h[i + 1 : i + hold + 1])) / entry - 1.0 >= TARGET else 0.0
            y[idx] = yy
        ycols[f"mfe_{hold}"] = y
    out = df.copy()
    for k, v in ycols.items():
        out[k] = v
    return out


GATES = {
    "thrust_break": lambda d: (
        (d.mom1 >= 2.0) & (d.vol_ratio >= 2.0) & (d.break_sig == 1) & (d.trend >= 1) & d.rsi.between(55, 75) & (d.atr_pct >= 2.0)
    ),
    "break_vol": lambda d: (
        (d.break_sig == 1) & (d.trend >= 1) & (d.vol_ratio >= 1.8) & (d.mom1 >= 1.0) & d.rsi.between(55, 72) & (d.atr_pct >= 2.0)
    ),
    "coil_bull": lambda d: (
        (d.trend == 2) & d.rsi.between(48, 58) & (d.ext_atr <= 1.0) & d.vol_ratio.between(0.9, 1.8) & (d.atr_pct >= 2.0) & d.mom5.between(-3, 4)
    ),
    "near52": lambda d: (
        (d.dist_52w_high >= -6) & (d.trend == 2) & (d.vol_ratio >= 1.2) & d.rsi.between(52, 68) & (d.atr_pct >= 2.0)
    ),
    "hi_atr_trend": lambda d: (
        (d.trend == 2) & (d.atr_pct >= 3.5) & (d.vol_ratio >= 1.2) & d.rsi.between(50, 70) & (d.ext_atr <= 2.5)
    ),
    "gap_cont": lambda d: (
        (d.gap_pct >= 1.2) & (d.mom1 >= 1.0) & (d.vol_ratio >= 1.6) & (d.trend >= 1) & d.rsi.between(52, 72)
    ),
    "mom20_power": lambda d: (
        (d.mom20 >= 8) & (d.trend == 2) & (d.vol_ratio >= 1.1) & (d.atr_pct >= 2.0) & d.rsi.between(55, 75)
    ),
    "squeeze_release": lambda d: (
        (d.trend == 2) & (d.compress <= 0.4) & (d.mom1 >= 0.8) & (d.vol_ratio >= 1.4) & (d.atr_pct >= 2.0) & d.rsi.between(50, 65)
    ),
    "pull_resume": lambda d: (
        (d.trend == 2) & d.rsi.between(42, 52) & (d.ext_atr <= 1.2) & (d.mom1 > 0) & (d.vol_ratio >= 1.1) & (d.atr_pct >= 2.0)
    ),
}


FEATS = [
    "trend",
    "rsi",
    "atr_pct",
    "vol_ratio",
    "ext_atr",
    "mom1",
    "mom5",
    "mom20",
    "mom60",
    "break_sig",
    "dist_52w_high",
    "dist_52w_low",
    "gap_pct",
    "compress",
]


def split_bf(df: pd.DataFrame):
    dates = np.sort(df.Date.unique())
    cut = dates[int(len(dates) * 0.70)]
    back = df[df.Date < cut].copy()
    fwd = df[df.Date >= cut].copy()
    return back, fwd, pd.Timestamp(cut)


def eval_gates(df: pd.DataFrame, holds: list[int]) -> list[dict]:
    back, fwd, cut = split_bf(df)
    print(f"cut {cut.date()} back={len(back):,} fwd={len(fwd):,}")
    rows = []
    for hold in holds:
        ycol = f"mfe_{hold}"
        b = back.dropna(subset=[ycol])
        f = fwd.dropna(subset=[ycol])
        base_b, base_f = float(b[ycol].mean()), float(f[ycol].mean())
        print(f"  hold={hold}d base back={base_b*100:.1f}% fwd={base_f*100:.1f}%")
        for gname, gfn in GATES.items():
            tb, tf = b.loc[gfn(b)], f.loc[gfn(f)]
            if len(tb) < MIN_TRAIN or len(tf) < MIN_FWD:
                continue
            tw, fw = float(tb[ycol].mean()), float(tf[ycol].mean())
            rows.append(
                {
                    "kind": "gate_mfe",
                    "gate": gname,
                    "hold": hold,
                    "back_n": len(tb),
                    "back_wr": round(tw, 4),
                    "fwd_n": len(tf),
                    "fwd_wr": round(fw, 4),
                    "wilson": round(wilson(int(tf[ycol].sum()), len(tf)), 4),
                    "hit80": tw >= TARGET_WR and fw >= TARGET_WR,
                    "base_fwd": round(base_f, 4),
                }
            )
    return rows


def eval_gbm(df: pd.DataFrame, holds: list[int]) -> list[dict]:
    back, fwd, _ = split_bf(df)
    rows = []
    for hold in holds:
        ycol = f"mfe_{hold}"
        b = back.dropna(subset=[ycol] + FEATS)
        f = fwd.dropna(subset=[ycol] + FEATS)
        if b[ycol].nunique() < 2:
            continue
        clf = CalibratedClassifierCV(
            HistGradientBoostingClassifier(
                max_depth=4, learning_rate=0.05, max_iter=200, min_samples_leaf=40, l2_regularization=0.1, random_state=SEED
            ),
            method="isotonic",
            cv=3,
        )
        clf.fit(b[FEATS], b[ycol].astype(int))
        p_b = clf.predict_proba(b[FEATS])[:, 1]
        p_f = clf.predict_proba(f[FEATS])[:, 1]
        yb, yf = b[ycol].astype(int).to_numpy(), f[ycol].astype(int).to_numpy()
        auc = roc_auc_score(yf, p_f) if len(np.unique(yf)) > 1 else float("nan")
        for thr in np.linspace(0.35, 0.95, 25):
            mb, mf = p_b >= thr, p_f >= thr
            nb, nf = int(mb.sum()), int(mf.sum())
            if nb < MIN_TRAIN or nf < MIN_FWD:
                continue
            tw = float(precision_score(yb, mb.astype(int), zero_division=0))
            fw = float(precision_score(yf, mf.astype(int), zero_division=0))
            rows.append(
                {
                    "kind": "gbm_mfe",
                    "gate": f"gbm_thr_{thr:.2f}",
                    "hold": hold,
                    "thr": float(thr),
                    "back_n": nb,
                    "back_wr": round(tw, 4),
                    "fwd_n": nf,
                    "fwd_wr": round(fw, 4),
                    "wilson": round(wilson(int(yf[mf].sum()), nf), 4),
                    "hit80": tw >= TARGET_WR and fw >= TARGET_WR,
                    "auc": round(float(auc), 4) if auc == auc else None,
                    "recall_fwd": round(float(recall_score(yf, mf.astype(int), zero_division=0)), 4),
                }
            )
        # top-k forward ranked by score, measure precision; calibrate k on back similarly
        for pct in (0.5, 1.0, 2.0):
            kb = max(MIN_TRAIN, int(len(p_b) * pct / 100))
            kf = max(MIN_FWD, int(len(p_f) * pct / 100))
            topb = np.argsort(p_b)[-kb:]
            topf = np.argsort(p_f)[-kf:]
            tw, fw = float(yb[topb].mean()), float(yf[topf].mean())
            rows.append(
                {
                    "kind": "gbm_topk",
                    "gate": f"top_{pct}pct",
                    "hold": hold,
                    "back_n": kb,
                    "back_wr": round(tw, 4),
                    "fwd_n": kf,
                    "fwd_wr": round(fw, 4),
                    "wilson": round(wilson(int(yf[topf].sum()), kf), 4),
                    "hit80": tw >= TARGET_WR and fw >= TARGET_WR,
                    "auc": round(float(auc), 4) if auc == auc else None,
                }
            )
        print(f"  GBM hold={hold}d AUC={auc:.3f} best_fwd={max((r['fwd_wr'] for r in rows if r['hold']==hold), default=0)*100:.1f}%")
    return rows


def eval_path(df: pd.DataFrame) -> list[dict]:
    """+10% before stop within hold — may raise WR vs pure MFE."""
    back, fwd, _ = split_bf(df)
    rows = []
    for stop_pct, hold in product([0.03, 0.05, 0.08, 0.10, 0.12], [10, 15, 20, 30, 45]):
        y = np.full(len(df), np.nan)
        for _, idx in df.groupby("Symbol").groups.items():
            idx = np.asarray(list(idx))
            sub = df.iloc[idx]
            y[idx] = path_label(
                sub.Close.to_numpy(float),
                sub.High.to_numpy(float),
                sub.Low.to_numpy(float),
                float(stop_pct),
                float(TARGET),
                int(hold),
            )
        work = df.copy()
        work["yp"] = y
        work = work.dropna(subset=["yp"])
        b, f = work[work.Date < back.Date.max()], work[work.Date >= fwd.Date.min()]
        # safer: use same cut
        dates = np.sort(df.Date.unique())
        cut = dates[int(len(dates) * 0.70)]
        b, f = work[work.Date < cut], work[work.Date >= cut]
        for gname, gfn in GATES.items():
            tb, tf = b.loc[gfn(b)], f.loc[gfn(f)]
            if len(tb) < MIN_TRAIN or len(tf) < MIN_FWD:
                continue
            tw, fw = float(tb.yp.mean()), float(tf.yp.mean())
            rows.append(
                {
                    "kind": "path",
                    "gate": gname,
                    "hold": hold,
                    "stop_pct": stop_pct,
                    "back_n": len(tb),
                    "back_wr": round(tw, 4),
                    "fwd_n": len(tf),
                    "fwd_wr": round(fw, 4),
                    "wilson": round(wilson(int(tf.yp.sum()), len(tf)), 4),
                    "hit80": tw >= TARGET_WR and fw >= TARGET_WR,
                    "rr": round(TARGET / stop_pct, 2),
                }
            )
    return rows


def eval_gate_and(df: pd.DataFrame, holds: list[int]) -> list[dict]:
    """Conjunction of two gates for higher precision."""
    back, fwd, _ = split_bf(df)
    names = list(GATES.keys())
    rows = []
    for hold in holds:
        ycol = f"mfe_{hold}"
        b = back.dropna(subset=[ycol])
        f = fwd.dropna(subset=[ycol])
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                g1, g2 = names[i], names[j]
                m_b = GATES[g1](b) & GATES[g2](b)
                m_f = GATES[g1](f) & GATES[g2](f)
                tb, tf = b.loc[m_b], f.loc[m_f]
                if len(tb) < MIN_TRAIN or len(tf) < MIN_FWD:
                    continue
                tw, fw = float(tb[ycol].mean()), float(tf[ycol].mean())
                rows.append(
                    {
                        "kind": "gate_and",
                        "gate": f"{g1}+{g2}",
                        "hold": hold,
                        "back_n": len(tb),
                        "back_wr": round(tw, 4),
                        "fwd_n": len(tf),
                        "fwd_wr": round(fw, 4),
                        "wilson": round(wilson(int(tf[ycol].sum()), len(tf)), 4),
                        "hit80": tw >= TARGET_WR and fw >= TARGET_WR,
                    }
                )
    return rows


def main():
    holds = [10, 15, 20, 30, 45, 60]
    panel = build_panel(180)
    panel = add_mfe_labels(panel, holds)

    print("\n=== Gate MFE search ===")
    gate_rows = eval_gates(panel, holds)
    print("\n=== Gate AND search ===")
    and_rows = eval_gate_and(panel, holds)
    print("\n=== GBM search ===")
    gbm_rows = eval_gbm(panel, holds)
    print("\n=== Path (+10% before stop) search ===")
    path_rows = eval_path(panel)

    all_rows = gate_rows + and_rows + gbm_rows + path_rows
    all_rows.sort(key=lambda r: (r.get("fwd_wr", 0), r.get("wilson", 0), r.get("fwd_n", 0)), reverse=True)
    hits = [r for r in all_rows if r.get("hit80")]
    usable = [r for r in hits if r.get("wilson", 0) >= 0.70 and r.get("fwd_n", 0) >= MIN_FWD]

    print(f"\nTotal candidates={len(all_rows)}  hit80={len(hits)}  usable(wilson>=70%)={len(usable)}")
    print("TOP 25 by forward WR:")
    for r in all_rows[:25]:
        mark = " ***80" if r.get("hit80") else ""
        print(
            f"  {r['kind']:10s} {str(r['gate'])[:32]:32s} H={r['hold']:<2} "
            f"fwd={r['fwd_wr']*100:5.1f}% wil={r.get('wilson',0)*100:4.1f}% n={r['fwd_n']:4d} "
            f"back={r['back_wr']*100:5.1f}% n={r['back_n']:4d}{mark}"
        )

    # Pick best usable scanner rule
    chosen = None
    if usable:
        usable.sort(key=lambda r: (r["fwd_wr"], r["wilson"], r["fwd_n"]), reverse=True)
        chosen = usable[0]
    elif hits:
        hits.sort(key=lambda r: (r.get("wilson", 0), r["fwd_wr"], r["fwd_n"]), reverse=True)
        chosen = hits[0]

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective": {
            "min_move": TARGET,
            "target_wr": TARGET_WR,
            "validation": "backtest(train early 70% dates) + forward(test latest 30% dates)",
            "freedom": "any hold, gate, model, path geometry",
        },
        "meta": {
            "symbols": int(panel.Symbol.nunique()),
            "rows": len(panel),
            "candidates": len(all_rows),
            "hit80": len(hits),
            "usable": len(usable),
            "verdict": "ACHIEVED" if usable else ("FRAGILE_80" if hits else "NOT_ACHIEVED"),
        },
        "chosen_scanner": chosen,
        "top_forward": all_rows[:30],
        "all_hit80": hits[:40],
    }
    (OUT / "scanner10pct_80wr_study.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# Full-freedom scanner: ≥10% move with ≥80% WR",
        "",
        f"Generated: {payload['created_utc']}",
        "",
        "## Protocol",
        "- **Backtest**: earliest ~70% of trading dates",
        "- **Forward test**: latest ~30% of dates (no peeking)",
        "- **Success**: tip WR ≥80% on **both** windows, forward n≥20, prefer Wilson≥70%",
        "- **Win**: +10% MFE within hold (or +10% before stop for path rules)",
        "",
        f"## Verdict: **{payload['meta']['verdict']}**",
        f"- Candidates scored: {len(all_rows)}",
        f"- Hit 80/80: {len(hits)}",
        f"- Usable (Wilson≥70%): {len(usable)}",
        "",
    ]
    if chosen:
        lines += [
            "## Chosen rule",
            f"```json\n{json.dumps(chosen, indent=2)}\n```",
            "",
        ]
    lines += [
        "## Top forward results",
        "| Kind | Gate | Hold | Fwd WR | Wilson | Fwd n | Back WR | Back n |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in all_rows[:20]:
        lines.append(
            f"| {r['kind']} | `{r['gate']}` | {r['hold']} | {r['fwd_wr']*100:.1f}% | "
            f"{r.get('wilson',0)*100:.1f}% | {r['fwd_n']} | {r['back_wr']*100:.1f}% | {r['back_n']} |"
        )
    (OUT / "scanner10pct_80wr_report.md").write_text("\n".join(lines) + "\n")
    print(f"\nVERDICT {payload['meta']['verdict']}")
    if chosen:
        print("CHOSEN", json.dumps(chosen, indent=2))
    print("wrote", OUT / "scanner10pct_80wr_report.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Non-linear full-freedom search: ≥+20% move with ≥80% tip WR.

Levers beyond linear technical gates:
  - Hold up to ~1y (30..252 sessions)
  - Index regime (Nifty bull / risk-on) as a hard gate
  - Cross-sectional ranks (RS vs Nifty, sector-proxy mom)
  - Extreme calibrated sparsity (P≥0.80..0.97)
  - Two-stage: high score + next-bar confirmation
  - Path wins: +20% before stop (raise WR, lower RR)

Protocol: backtest = earliest 70% dates; forward = latest 30%.
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scanners" / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)
TARGET = 0.20
TARGET_WR = 0.80
MIN_TRAIN = 30
MIN_FWD = 20
SEED = 42
HOLDS = [30, 45, 60, 90, 120, 150, 180, 252]


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


def nifty_regime() -> pd.DataFrame:
    n = yf.download("^NSEI", period="5y", interval="1d", progress=False, auto_adjust=False)
    if isinstance(n.columns, pd.MultiIndex):
        # yfinance may order levels as (Price, Ticker)
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
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(c.index).tz_localize(None),
            "nifty": c.values,
            "nifty_bull": ((c > ema50) & (ema50 > ema200)).astype(int).values,
            "nifty_mom20": (c.pct_change(20) * 100).values,
            "nifty_mom60": (c.pct_change(60) * 100).values,
        }
    ).dropna()
    return out


def build_panel(n_syms: int = 180) -> pd.DataFrame:
    syms = sorted(pd.read_csv(ROOT / "fno_adv.csv").Symbol.astype(str).str.strip().str.upper().unique())
    rng = np.random.default_rng(SEED)
    if len(syms) > n_syms:
        syms = sorted(rng.choice(syms, size=n_syms, replace=False).tolist())
    tick = [f"{s}.NS" for s in syms]
    print(f"download {len(tick)} × 5y + Nifty")
    raw = yf.download(tick, period="5y", interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=False)
    reg = nifty_regime()
    pieces = []
    for sym, t in zip(syms, tick):
        try:
            bar = raw[t].dropna(how="all")
            if len(bar) < 300:
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
            idx = pd.to_datetime(bar.index).tz_localize(None)
            df = pd.DataFrame(
                {
                    "Symbol": sym,
                    "Close": c.to_numpy(),
                    "High": h.to_numpy(),
                    "Low": l.to_numpy(),
                    "Open": o.to_numpy(),
                    "ATR": a.to_numpy(),
                    "trend": trend,
                    "rsi": r.to_numpy(),
                    "atr_pct": (a / c * 100).to_numpy(),
                    "vol_ratio": (v / volma).to_numpy(),
                    "ext_atr": ((c - ema20).abs() / a.replace(0, np.nan)).to_numpy(),
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
                    "dow": idx.dayofweek,
                    "Date": idx,
                }
            ).replace([np.inf, -np.inf], np.nan)
            df = df.merge(reg, on="Date", how="left")
            df["rs20"] = df["mom20"] - df["nifty_mom20"]
            df["rs60"] = df["mom60"] - df["nifty_mom60"]
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            if len(df) >= 260:
                pieces.append(df.reset_index(drop=True))
        except Exception as exc:
            print(f"  skip {sym}: {exc}")
            continue
    if not pieces:
        raise RuntimeError("empty panel — Yahoo/Nifty merge failed")
    out = pd.concat(pieces, ignore_index=True)
    # cross-sectional ranks per date
    out["rank_mom20"] = out.groupby("Date")["mom20"].rank(pct=True)
    out["rank_rs20"] = out.groupby("Date")["rs20"].rank(pct=True)
    out["rank_atr"] = out.groupby("Date")["atr_pct"].rank(pct=True)
    out["rank_vol"] = out.groupby("Date")["vol_ratio"].rank(pct=True)
    print(f"panel rows={len(out):,} syms={out.Symbol.nunique()}")
    return out


def add_mfe_labels(df: pd.DataFrame, holds: list[int]) -> pd.DataFrame:
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
    "hi_atr_trend": lambda d: (
        (d.trend == 2) & (d.atr_pct >= 3.5) & (d.vol_ratio >= 1.2) & d.rsi.between(50, 70) & (d.ext_atr <= 2.5)
    ),
    "near52": lambda d: (
        (d.dist_52w_high >= -6) & (d.trend == 2) & (d.vol_ratio >= 1.2) & d.rsi.between(52, 68) & (d.atr_pct >= 2.0)
    ),
    "mom20_power": lambda d: (
        (d.mom20 >= 8) & (d.trend == 2) & (d.vol_ratio >= 1.1) & (d.atr_pct >= 2.0) & d.rsi.between(55, 75)
    ),
    "rs_leader": lambda d: (
        (d.rank_rs20 >= 0.90) & (d.trend == 2) & (d.nifty_bull == 1) & (d.atr_pct >= 2.0) & d.rsi.between(50, 72)
    ),
    "regime_bull_break": lambda d: (
        (d.nifty_bull == 1) & (d.break_sig == 1) & (d.trend == 2) & (d.vol_ratio >= 1.5) & (d.mom1 >= 1.0) & (d.atr_pct >= 2.5)
    ),
    "xsec_thrust": lambda d: (
        (d.rank_mom20 >= 0.95) & (d.rank_vol >= 0.80) & (d.trend >= 1) & (d.nifty_bull == 1) & (d.atr_pct >= 2.0)
    ),
    "long_mom": lambda d: (
        (d.mom60 >= 15) & (d.mom120 >= 20) & (d.trend == 2) & (d.rs60 >= 5) & (d.nifty_bull == 1)
    ),
    "squeeze_release": lambda d: (
        (d.trend == 2) & (d.compress <= 0.4) & (d.mom1 >= 0.8) & (d.vol_ratio >= 1.4) & (d.atr_pct >= 2.0) & d.rsi.between(50, 65)
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
    "mom120",
    "break_sig",
    "dist_52w_high",
    "dist_52w_low",
    "gap_pct",
    "compress",
    "nifty_bull",
    "nifty_mom20",
    "rs20",
    "rs60",
    "rank_mom20",
    "rank_rs20",
    "rank_atr",
    "rank_vol",
    "month",
]


def split_bf(df: pd.DataFrame):
    dates = np.sort(df.Date.unique())
    cut = dates[int(len(dates) * 0.70)]
    back = df[df.Date < cut].copy()
    fwd = df[df.Date >= cut].copy()
    return back, fwd, pd.Timestamp(cut)


def eval_gates(df: pd.DataFrame) -> list[dict]:
    back, fwd, cut = split_bf(df)
    print(f"cut {cut.date()} back={len(back):,} fwd={len(fwd):,}")
    rows = []
    for hold in HOLDS:
        ycol = f"mfe_{hold}"
        b = back.dropna(subset=[ycol])
        f = fwd.dropna(subset=[ycol])
        base_b, base_f = float(b[ycol].mean()), float(f[ycol].mean())
        print(f"  hold={hold}d base back={base_b*100:.1f}% fwd={base_f*100:.1f}%")
        for gname, gfn in GATES.items():
            for regime_only in (False, True):
                mb = gfn(b)
                mf = gfn(f)
                if regime_only:
                    mb = mb & (b.nifty_bull == 1)
                    mf = mf & (f.nifty_bull == 1)
                tb, tf = b.loc[mb], f.loc[mf]
                if len(tb) < MIN_TRAIN or len(tf) < MIN_FWD:
                    continue
                tw, fw = float(tb[ycol].mean()), float(tf[ycol].mean())
                rows.append(
                    {
                        "kind": "gate_mfe",
                        "gate": gname + ("|nifty_bull" if regime_only else ""),
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


def eval_gbm(df: pd.DataFrame, regime_filter: bool = False) -> list[dict]:
    back, fwd, _ = split_bf(df)
    rows = []
    tag = "gbm_regime" if regime_filter else "gbm_mfe"
    for hold in HOLDS:
        ycol = f"mfe_{hold}"
        b = back.dropna(subset=[ycol] + FEATS)
        f = fwd.dropna(subset=[ycol] + FEATS)
        if regime_filter:
            b = b[b.nifty_bull == 1]
            f = f[f.nifty_bull == 1]
        if len(b) < 500 or b[ycol].nunique() < 2 or len(f) < 100:
            continue
        clf = CalibratedClassifierCV(
            HistGradientBoostingClassifier(
                max_depth=4, learning_rate=0.05, max_iter=220, min_samples_leaf=40, l2_regularization=0.1, random_state=SEED
            ),
            method="isotonic",
            cv=3,
        )
        clf.fit(b[FEATS], b[ycol].astype(int))
        p_b = clf.predict_proba(b[FEATS])[:, 1]
        p_f = clf.predict_proba(f[FEATS])[:, 1]
        yb, yf = b[ycol].astype(int).to_numpy(), f[ycol].astype(int).to_numpy()
        auc = roc_auc_score(yf, p_f) if len(np.unique(yf)) > 1 else float("nan")
        best_fwd = 0.0
        for thr in np.linspace(0.40, 0.97, 30):
            mb, mf = p_b >= thr, p_f >= thr
            nb, nf = int(mb.sum()), int(mf.sum())
            if nb < MIN_TRAIN or nf < MIN_FWD:
                continue
            tw = float(precision_score(yb, mb.astype(int), zero_division=0))
            fw = float(precision_score(yf, mf.astype(int), zero_division=0))
            best_fwd = max(best_fwd, fw)
            rows.append(
                {
                    "kind": tag,
                    "gate": f"thr_{thr:.2f}",
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
        for pct in (0.25, 0.5, 1.0):
            kb = max(MIN_TRAIN, int(len(p_b) * pct / 100))
            kf = max(MIN_FWD, int(len(p_f) * pct / 100))
            if kb > len(p_b) or kf > len(p_f):
                continue
            topb = np.argsort(p_b)[-kb:]
            topf = np.argsort(p_f)[-kf:]
            tw, fw = float(yb[topb].mean()), float(yf[topf].mean())
            best_fwd = max(best_fwd, fw)
            rows.append(
                {
                    "kind": tag + "_topk",
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
        print(f"  {tag} hold={hold}d AUC={auc:.3f} best_fwd={best_fwd*100:.1f}%")
    return rows


def eval_two_stage(df: pd.DataFrame) -> list[dict]:
    """High score day-0 + next-day follow-through confirmation (non-linear path)."""
    back, fwd, _ = split_bf(df)
    rows = []
    for hold in [60, 90, 120, 180, 252]:
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
        for split_df, name in ((b, "back"), (f, "fwd")):
            pass
        p_b = clf.predict_proba(b[FEATS])[:, 1]
        p_f = clf.predict_proba(f[FEATS])[:, 1]
        b = b.assign(_p=p_b, _i=np.arange(len(b)))
        f = f.assign(_p=p_f, _i=np.arange(len(f)))

        def confirm_mask(part: pd.DataFrame, thr: float) -> pd.Series:
            # tip only if score≥thr AND next session close > entry (follow-through)
            m = part._p >= thr
            # next-day close confirmation within symbol
            nxt = part.groupby("Symbol")["Close"].shift(-1)
            conf = nxt > part.Close
            return m & conf.fillna(False) & (part.nifty_bull == 1)

        yb = b[ycol].astype(int)
        yf = f[ycol].astype(int)
        for thr in np.linspace(0.55, 0.95, 17):
            mb, mf = confirm_mask(b, thr), confirm_mask(f, thr)
            nb, nf = int(mb.sum()), int(mf.sum())
            if nb < MIN_TRAIN or nf < MIN_FWD:
                continue
            tw = float(yb[mb].mean())
            fw = float(yf[mf].mean())
            rows.append(
                {
                    "kind": "two_stage",
                    "gate": f"p>={thr:.2f}+nxt_up+nifty",
                    "hold": hold,
                    "thr": float(thr),
                    "back_n": nb,
                    "back_wr": round(tw, 4),
                    "fwd_n": nf,
                    "fwd_wr": round(fw, 4),
                    "wilson": round(wilson(int(yf[mf].sum()), nf), 4),
                    "hit80": tw >= TARGET_WR and fw >= TARGET_WR,
                }
            )
        if rows:
            best = max((r["fwd_wr"] for r in rows if r["hold"] == hold), default=0)
            print(f"  two_stage hold={hold}d best_fwd={best*100:.1f}%")
    return rows


def eval_path(df: pd.DataFrame) -> list[dict]:
    """+20% before stop — WR inflation lever (RR may be <1)."""
    from numba import njit

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

    rows = []
    dates = np.sort(df.Date.unique())
    cut = dates[int(len(dates) * 0.70)]
    for stop_pct, hold in product([0.05, 0.08, 0.10, 0.12, 0.15], [30, 60, 90, 120]):
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
        work = df.assign(yp=y).dropna(subset=["yp"])
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
    if rows:
        best = max(r["fwd_wr"] for r in rows)
        print(f"  path best_fwd={best*100:.1f}% hit80={sum(1 for r in rows if r['hit80'])}")
    return rows


def main():
    panel = build_panel(180)
    panel = add_mfe_labels(panel, HOLDS)

    print("\n=== Gate / regime MFE ===")
    gate_rows = eval_gates(panel)
    print("\n=== GBM + RS/regime features ===")
    gbm_rows = eval_gbm(panel, regime_filter=False)
    print("\n=== GBM trained only in Nifty-bull regime ===")
    gbm_reg = eval_gbm(panel, regime_filter=True)
    print("\n=== Two-stage confirmation ===")
    two = eval_two_stage(panel)
    print("\n=== Path +20% before stop ===")
    path_rows = eval_path(panel)

    all_rows = gate_rows + gbm_rows + gbm_reg + two + path_rows
    all_rows.sort(key=lambda r: (r.get("fwd_wr", 0), r.get("wilson", 0), r.get("fwd_n", 0)), reverse=True)
    hits = [r for r in all_rows if r.get("hit80")]
    usable = [r for r in hits if r.get("wilson", 0) >= 0.70 and r.get("fwd_n", 0) >= MIN_FWD]

    print(f"\nTotal candidates={len(all_rows)}  hit80={len(hits)}  usable={len(usable)}")
    print("TOP 30 by forward WR:")
    for r in all_rows[:30]:
        mark = " ***80" if r.get("hit80") else ""
        print(
            f"  {r['kind']:14s} {str(r['gate'])[:36]:36s} H={r['hold']:<3} "
            f"fwd={r['fwd_wr']*100:5.1f}% wil={r.get('wilson',0)*100:4.1f}% n={r['fwd_n']:4d} "
            f"back={r['back_wr']*100:5.1f}% n={r['back_n']:4d}{mark}"
        )

    chosen = None
    if usable:
        usable.sort(key=lambda r: (r["fwd_wr"], r["wilson"], r["fwd_n"]), reverse=True)
        chosen = usable[0]
    elif hits:
        hits.sort(key=lambda r: (r.get("wilson", 0), r["fwd_wr"], r["fwd_n"]), reverse=True)
        chosen = hits[0]

    # best by hold for design memo
    by_hold = {}
    for h in HOLDS:
        cand = [r for r in all_rows if r["hold"] == h]
        by_hold[h] = cand[0] if cand else None

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective": {
            "min_move": TARGET,
            "target_wr": TARGET_WR,
            "validation": "backtest early 70% + forward latest 30%",
            "nonlinear_levers": [
                "hold_to_252d",
                "nifty_regime_gate",
                "cross_sectional_ranks",
                "extreme_calibration_sparsity",
                "two_stage_confirmation",
                "path_before_stop",
            ],
        },
        "meta": {
            "symbols": int(panel.Symbol.nunique()),
            "rows": len(panel),
            "candidates": len(all_rows),
            "hit80": len(hits),
            "usable": len(usable),
            "verdict": "ACHIEVED_ROBUST" if usable else ("FRAGILE_80" if hits else "NOT_ACHIEVED"),
        },
        "chosen_scanner": chosen,
        "best_by_hold": {str(k): v for k, v in by_hold.items()},
        "top_forward": all_rows[:40],
        "all_hit80": hits[:50],
    }
    (OUT / "scanner20pct_80wr_study.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# Non-linear design search: ≥20% move @ ≥80% tip WR",
        "",
        f"Generated: {payload['created_utc'][:10]}",
        "",
        "## Protocol",
        "- Backtest earliest ~70% dates / forward latest ~30%",
        "- Win: `max(high)` within hold ≥ entry × **1.20** (or +20% before stop for path rules)",
        "- Bar: ≥80% on **both** windows, forward n≥20, prefer Wilson≥70%",
        "",
        "### Non-linear levers tested",
        "1. Hold length up to **252** sessions (~1y)",
        "2. Hard **Nifty bull** regime filter",
        "3. Cross-sectional ranks (RS vs Nifty, mom/vol)",
        "4. Extreme calibrated GBM sparsity",
        "5. Two-stage: high P + next-bar confirmation + regime",
        "6. Path geometry: +20% before stop (WR↑, RR may fall)",
        "",
        f"## Verdict: **{payload['meta']['verdict']}**",
        f"- Candidates: {len(all_rows)} | hit80: {len(hits)} | usable: {len(usable)}",
        "",
        "## Best forward WR by hold",
        "| Hold | Kind | Gate | Fwd WR | Wilson | n | Back WR |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for h in HOLDS:
        r = by_hold.get(h)
        if not r:
            continue
        lines.append(
            f"| {h} | {r['kind']} | `{r['gate']}` | {r['fwd_wr']*100:.1f}% | "
            f"{r.get('wilson',0)*100:.1f}% | {r['fwd_n']} | {r['back_wr']*100:.1f}% |"
        )
    if chosen:
        lines += ["", "## Chosen (if any)", f"```json\n{json.dumps(chosen, indent=2)}\n```"]
    lines += [
        "",
        "## Top 20 forward",
        "| Kind | Gate | Hold | Fwd | Wilson | n | Back | n |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in all_rows[:20]:
        lines.append(
            f"| {r['kind']} | `{r['gate']}` | {r['hold']} | {r['fwd_wr']*100:.1f}% | "
            f"{r.get('wilson',0)*100:.1f}% | {r['fwd_n']} | {r['back_wr']*100:.1f}% | {r['back_n']} |"
        )
    (OUT / "scanner20pct_80wr_report.md").write_text("\n".join(lines) + "\n")
    print(f"\nVERDICT {payload['meta']['verdict']}")
    if chosen:
        print("CHOSEN", json.dumps(chosen, indent=2))
    print("wrote", OUT / "scanner20pct_80wr_report.md")


if __name__ == "__main__":
    main()

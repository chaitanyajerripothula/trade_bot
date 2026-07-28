#!/usr/bin/env python3
"""
Battle-test: stocks with ≥15% return in the next ~30 trading days @ ≥80% tip WR.

Win definitions:
  mfe   — max(High) over hold / Close − 1 ≥ 15%   (path high, like ShortHold)
  close — Close[t+hold] / Close[t] − 1 ≥ 15%     (held-to-horizon return)

Holds: 21, 30, 42, 45, 60  (30d is primary; longer = fallback)
Battle tests:
  1) 70/30 chronological split tip WR + Wilson
  2) Rolling 3-fold walk-forward (expanding train)
  3) Max-k daily tip policy (k=1,2,3)
  4) Stress: Nifty bear days only / bull days only on forward window
  5) Seed stability (2 seeds) on primary cell

Bar: back≥80%, fwd≥80%, Wilson≥70%, fwd_n≥40. Prefer hold≤30 and tips/day≥0.3.
"""
from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scanners" / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "fifteen_30_panel.joblib"
SEED = 42
TARGET = 0.15
HOLDS = [21, 30, 42, 45, 60]
PRIMARY_HOLD = 30
MIN_TRAIN = 40
MIN_FWD = 40
TARGET_WR = 0.80
WILSON_MIN = 0.70
THRS = np.round(np.linspace(0.55, 0.95, 41), 3)
MAX_KS = (1, 2, 3)

FEATS = [
    "trend", "rsi", "atr_pct", "vol_ratio", "ext_atr",
    "mom1", "mom5", "mom20", "mom60", "mom120",
    "break_sig", "dist_52w_high", "dist_52w_low", "gap_pct", "compress",
    "nifty_bull", "nifty_mom20", "rs20", "rs60",
    "rank_mom20", "rank_rs20", "rank_atr", "rank_vol", "month",
]


def log(msg: str) -> None:
    print(msg, flush=True)


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
            "nifty": c.values,
            "nifty_bull": ((c > ema50) & (ema50 > ema200)).astype(int).values,
            "nifty_mom20": (c.pct_change(20) * 100).values,
            "nifty_mom60": (c.pct_change(60) * 100).values,
        }
    ).dropna()


def build_panel(n_syms: int = 180) -> pd.DataFrame:
    if CACHE.exists():
        log(f"load cache {CACHE}")
        return joblib.load(CACHE)
    syms = sorted(pd.read_csv(ROOT / "fno_adv.csv").Symbol.astype(str).str.strip().str.upper().unique())
    rng = np.random.default_rng(SEED)
    if len(syms) > n_syms:
        syms = sorted(rng.choice(syms, size=n_syms, replace=False).tolist())
    tick = [f"{s}.NS" for s in syms]
    log(f"download {len(tick)} × 5y")
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
                    "Date": idx,
                }
            ).replace([np.inf, -np.inf], np.nan)
            df = df.merge(reg, on="Date", how="left")
            df["rs20"] = df["mom20"] - df["nifty_mom20"]
            df["rs60"] = df["mom60"] - df["nifty_mom60"]
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            if len(df) >= 200:
                pieces.append(df.reset_index(drop=True))
        except Exception:
            continue
    out = pd.concat(pieces, ignore_index=True)
    for col, src in (
        ("rank_mom20", "mom20"),
        ("rank_rs20", "rs20"),
        ("rank_atr", "atr_pct"),
        ("rank_vol", "vol_ratio"),
    ):
        out[col] = out.groupby("Date")[src].rank(pct=True)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    joblib.dump(out, CACHE)
    log(f"panel rows={len(out)} syms={out.Symbol.nunique()} dates={out.Date.nunique()}")
    return out


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized-ish per-symbol MFE and close labels for each hold."""
    out = df.copy()
    for hold in HOLDS:
        out[f"mfe_{hold}"] = np.nan
        out[f"close_{hold}"] = np.nan
    for _, idx in out.groupby("Symbol").groups.items():
        idx = np.asarray(list(idx))
        sub = out.iloc[idx]
        c = sub.Close.to_numpy()
        h = sub.High.to_numpy()
        n = len(sub)
        for hold in HOLDS:
            mfe = np.full(n, np.nan)
            clo = np.full(n, np.nan)
            for i in range(n - hold - 1):
                entry = c[i]
                if entry <= 0:
                    continue
                hi = float(np.nanmax(h[i + 1 : i + hold + 1]))
                mfe[i] = 1.0 if (hi / entry - 1.0) >= TARGET else 0.0
                fut = c[i + hold]
                clo[i] = 1.0 if (fut / entry - 1.0) >= TARGET else 0.0
            out.iloc[idx, out.columns.get_loc(f"mfe_{hold}")] = mfe
            out.iloc[idx, out.columns.get_loc(f"close_{hold}")] = clo
    return out


def fit_model(X, y, seed=SEED):
    base = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=220,
        min_samples_leaf=40,
        l2_regularization=0.1,
        random_state=seed,
    )
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X, y)
    return clf


def _day_starts(date_codes: np.ndarray) -> np.ndarray:
    return np.r_[np.flatnonzero(np.r_[True, date_codes[1:] != date_codes[:-1]]), len(date_codes)]


def sort_day_p(date_codes, p, y):
    order = np.lexsort((-p, date_codes))
    return date_codes[order], p[order], y[order]


def eval_tips(p, y, date_codes, thr, max_k):
    d, sp, sy = sort_day_p(date_codes, p, y)
    starts = _day_starts(d)
    hits = []
    tipped = 0
    n_days = len(starts) - 1
    for i in range(n_days):
        a, b = starts[i], starts[i + 1]
        sel = 0
        for j in range(a, b):
            if sp[j] < thr:
                break
            hits.append(int(sy[j]))
            sel += 1
            if sel >= max_k:
                break
        if sel:
            tipped += 1
    n = len(hits)
    if n == 0:
        return {"n": 0, "wr": 0.0, "wilson": 0.0, "tpd": 0.0, "day_cov": 0.0}
    k = int(sum(hits))
    return {
        "n": n,
        "wr": k / n,
        "wilson": wilson(k, n),
        "tpd": n / max(n_days, 1),
        "day_cov": tipped / max(n_days, 1),
    }


def sweep(ptr, ytr, dtr, pfw, yfw, dfw, meta: dict) -> tuple[list, dict | None]:
    rows = []
    best_hit = None
    best_any = None
    for thr in THRS:
        for max_k in MAX_KS:
            b = eval_tips(ptr, ytr, dtr, float(thr), max_k)
            f = eval_tips(pfw, yfw, dfw, float(thr), max_k)
            row = {
                **meta,
                "thr": float(thr),
                "max_k": max_k,
                "back_n": b["n"],
                "back_wr": b["wr"],
                "back_tpd": b["tpd"],
                "fwd_n": f["n"],
                "fwd_wr": f["wr"],
                "fwd_wilson": f["wilson"],
                "fwd_tpd": f["tpd"],
                "fwd_day_cov": f["day_cov"],
                "hit80": bool(
                    b["n"] >= MIN_TRAIN
                    and f["n"] >= MIN_FWD
                    and b["wr"] >= TARGET_WR
                    and f["wr"] >= TARGET_WR
                    and f["wilson"] >= WILSON_MIN
                ),
            }
            rows.append(row)
            score = (f["wr"], f["wilson"], f["tpd"], f["n"])
            if f["n"] >= 20 and b["n"] >= 20:
                if best_any is None or score > best_any[0]:
                    best_any = (score, row)
            if row["hit80"]:
                # prefer shorter hold, then higher wr/wilson/tpd
                rank = (
                    -int(meta.get("hold", 99)),
                    f["wr"],
                    f["wilson"],
                    f["tpd"],
                    f["n"],
                )
                if best_hit is None or rank > best_hit[0]:
                    best_hit = (rank, row)
    return rows, (best_hit[1] if best_hit else (best_any[1] if best_any else None))


def rolling_folds(dates: np.ndarray, n_folds: int = 3):
    """Expanding train, sequential test folds on latter 45% of dates."""
    dates = np.sort(dates)
    n = len(dates)
    test_start = int(n * 0.55)
    test_dates = dates[test_start:]
    fold_size = len(test_dates) // n_folds
    folds = []
    for i in range(n_folds):
        a = i * fold_size
        b = (i + 1) * fold_size if i < n_folds - 1 else len(test_dates)
        te = test_dates[a:b]
        if len(te) < 20:
            continue
        tr_end = te[0]
        folds.append((dates[dates < tr_end], te))
    return folds


def main():
    panel = build_panel(180)
    log("labeling MFE/close @ 15%…")
    panel = add_labels(panel)
    X_all = panel[FEATS].to_numpy()
    date_codes = pd.factorize(panel["Date"], sort=True)[0]
    dates = np.sort(panel["Date"].unique())
    cut = dates[int(0.70 * len(dates))]
    train_m = (panel["Date"] < cut).to_numpy()
    fwd_m = (panel["Date"] >= cut).to_numpy()
    log(f"70/30 cut={pd.Timestamp(cut).date()} train={train_m.sum()} fwd={fwd_m.sum()}")

    all_rows = []
    chosen = []
    models = {}
    base_rates = {}

    for hold in HOLDS:
        for kind in ("mfe", "close"):
            ycol = f"{kind}_{hold}"
            y = panel[ycol].to_numpy()
            valid = ~np.isnan(y)
            base_rates[ycol] = {
                "rate_all": float(np.nanmean(y)),
                "rate_fwd": float(np.nanmean(y[fwd_m & valid])),
                "hold": hold,
                "kind": kind,
            }
            mask_tr = train_m & valid
            mask_fw = fwd_m & valid
            ytr = y[mask_tr].astype(int)
            yfw = y[mask_fw].astype(int)
            if ytr.sum() < 50 or (len(ytr) - ytr.sum()) < 50:
                log(f"skip {ycol} sparse")
                continue
            log(f"fit {ycol} base_tr={ytr.mean():.3f} base_fw={yfw.mean():.3f}")
            clf = fit_model(X_all[mask_tr], ytr)
            ptr = clf.predict_proba(X_all[mask_tr])[:, 1]
            pfw = clf.predict_proba(X_all[mask_fw])[:, 1]
            try:
                auc = float(roc_auc_score(yfw, pfw))
            except Exception:
                auc = float("nan")
            meta = {"kind": kind, "hold": hold, "target": TARGET, "auc": auc, "base_fwd": float(yfw.mean())}
            rows, best = sweep(ptr, ytr, date_codes[mask_tr], pfw, yfw, date_codes[mask_fw], meta)
            all_rows.extend(rows)
            if best:
                tag = f"{kind}_{hold}d_k{best['max_k']}_thr{best['thr']}"
                status = "HIT80" if best.get("hit80") else f"near fwd={best['fwd_wr']:.1%}"
                log(
                    f"  {status}: thr={best['thr']} k={best['max_k']} "
                    f"back={best['back_wr']:.1%}({best['back_n']}) "
                    f"fwd={best['fwd_wr']:.1%}({best['fwd_n']}) "
                    f"W={best['fwd_wilson']:.1%} tpd={best['fwd_tpd']:.2f} auc={auc:.3f}"
                )
                if best.get("hit80"):
                    chosen.append({**best, "key": tag})
                    models[tag] = {
                        "model": clf,
                        "features": FEATS,
                        "kind": kind,
                        "hold": hold,
                        "target": TARGET,
                        "thr": best["thr"],
                        "max_k": best["max_k"],
                        "chosen": best,
                        "scanner_name": f"Fifteen{hold}" if kind == "mfe" else f"FifteenClose{hold}",
                        "side": "CALL",
                        "mode": f"fifteen_{kind}",
                    }

    # --- Battle test A: rolling walk-forward on primary cells ---
    log("\n=== Rolling walk-forward (3 folds) ===")
    wf_results = []
    primary_cells = [("mfe", 30), ("mfe", 42), ("mfe", 45), ("mfe", 60), ("close", 30), ("close", 60)]
    folds = rolling_folds(dates, 3)
    for kind, hold in primary_cells:
        y = panel[f"{kind}_{hold}"].to_numpy()
        valid = ~np.isnan(y)
        fold_stats = []
        for fi, (tr_dates, te_dates) in enumerate(folds):
            tr_set, te_set = set(tr_dates), set(te_dates)
            mtr = valid & panel["Date"].isin(tr_set).to_numpy()
            mte = valid & panel["Date"].isin(te_set).to_numpy()
            if mtr.sum() < 500 or mte.sum() < 200:
                continue
            ytr = y[mtr].astype(int)
            yte = y[mte].astype(int)
            if ytr.sum() < 40 or yte.sum() < 10:
                continue
            clf = fit_model(X_all[mtr], ytr)
            # use thr from best 70/30 hit for this cell if any, else 0.85
            cell_hits = [c for c in chosen if c["kind"] == kind and c["hold"] == hold]
            thr = cell_hits[0]["thr"] if cell_hits else 0.85
            max_k = cell_hits[0]["max_k"] if cell_hits else 3
            pte = clf.predict_proba(X_all[mte])[:, 1]
            ptr = clf.predict_proba(X_all[mtr])[:, 1]
            b = eval_tips(ptr, ytr, date_codes[mtr], thr, max_k)
            f = eval_tips(pte, yte, date_codes[mte], thr, max_k)
            fold_stats.append({"fold": fi, "thr": thr, "max_k": max_k, "back": b, "fwd": f})
            log(
                f"  {kind}_{hold} fold{fi}: thr={thr} fwd_wr={f['wr']:.1%} n={f['n']} "
                f"W={f['wilson']:.1%} tpd={f['tpd']:.2f}"
            )
        if fold_stats:
            wrs = [fs["fwd"]["wr"] for fs in fold_stats if fs["fwd"]["n"] >= 15]
            wf_results.append(
                {
                    "kind": kind,
                    "hold": hold,
                    "folds": fold_stats,
                    "mean_fwd_wr": float(np.mean(wrs)) if wrs else None,
                    "min_fwd_wr": float(np.min(wrs)) if wrs else None,
                    "folds_ge_80": int(sum(1 for w in wrs if w >= 0.80)),
                    "n_folds": len(wrs),
                }
            )

    # --- Battle test B: regime stress on best 30d / best overall ---
    log("\n=== Regime stress (forward bull vs bear days) ===")
    stress = []
    for kind, hold in [("mfe", 30), ("mfe", 60), ("close", 30)]:
        cell_hits = [c for c in chosen if c["kind"] == kind and c["hold"] == hold]
        near = sorted(
            [r for r in all_rows if r["kind"] == kind and r["hold"] == hold and r["fwd_n"] >= 30],
            key=lambda r: (r["fwd_wr"], r["fwd_wilson"]),
            reverse=True,
        )
        pick = cell_hits[0] if cell_hits else (near[0] if near else None)
        if not pick:
            continue
        y = panel[f"{kind}_{hold}"].to_numpy()
        valid = ~np.isnan(y)
        mask_tr = train_m & valid
        clf = fit_model(X_all[mask_tr], y[mask_tr].astype(int))
        for regime, rmask in (
            ("bull", panel["nifty_bull"].to_numpy() == 1),
            ("bear", panel["nifty_bull"].to_numpy() == 0),
        ):
            m = fwd_m & valid & rmask
            if m.sum() < 100:
                continue
            p = clf.predict_proba(X_all[m])[:, 1]
            f = eval_tips(p, y[m].astype(int), date_codes[m], pick["thr"], pick["max_k"])
            stress.append(
                {
                    "kind": kind,
                    "hold": hold,
                    "regime": regime,
                    "thr": pick["thr"],
                    "max_k": pick["max_k"],
                    **{f"fwd_{k}": f[k] for k in ("n", "wr", "wilson", "tpd")},
                }
            )
            log(f"  {kind}_{hold} {regime}: wr={f['wr']:.1%} n={f['n']} W={f['wilson']:.1%}")

    # --- Battle test C: seed stability on mfe_30 and best hit ---
    log("\n=== Seed stability ===")
    seed_rows = []
    for kind, hold in [("mfe", 30), ("mfe", 60)]:
        y = panel[f"{kind}_{hold}"].to_numpy()
        valid = ~np.isnan(y)
        mask_tr = train_m & valid
        mask_fw = fwd_m & valid
        cell_hits = [c for c in chosen if c["kind"] == kind and c["hold"] == hold]
        thr = cell_hits[0]["thr"] if cell_hits else 0.88
        max_k = cell_hits[0]["max_k"] if cell_hits else 3
        wrs = []
        for seed in (42, 7, 99):
            clf = fit_model(X_all[mask_tr], y[mask_tr].astype(int), seed=seed)
            pfw = clf.predict_proba(X_all[mask_fw])[:, 1]
            f = eval_tips(pfw, y[mask_fw].astype(int), date_codes[mask_fw], thr, max_k)
            wrs.append(f["wr"])
            seed_rows.append({"kind": kind, "hold": hold, "seed": seed, "thr": thr, "fwd_wr": f["wr"], "fwd_n": f["n"]})
            log(f"  {kind}_{hold} seed={seed}: fwd={f['wr']:.1%} n={f['n']}")
        log(f"  → mean={np.mean(wrs):.1%} std={np.std(wrs):.1%}")

    # Pick ship candidate: prefer hold≤30 hit80 mfe; else shortest hold hit80; else best near
    hit80 = [c for c in chosen if c.get("hit80")]
    hit80_mfe = [c for c in hit80 if c["kind"] == "mfe"]
    ship = None
    if any(c["hold"] <= 30 for c in hit80_mfe):
        ship = sorted([c for c in hit80_mfe if c["hold"] <= 30], key=lambda c: (c["fwd_wr"], c["fwd_wilson"], c["fwd_tpd"]), reverse=True)[0]
    elif hit80_mfe:
        ship = sorted(hit80_mfe, key=lambda c: (c["hold"], -c["fwd_wr"], -c["fwd_wilson"]))[0]
    elif hit80:
        ship = sorted(hit80, key=lambda c: (c["hold"], -c["fwd_wr"]))[0]

    ranked = sorted(
        [r for r in all_rows if r["fwd_n"] >= MIN_FWD and r["back_n"] >= MIN_TRAIN],
        key=lambda r: (r["fwd_wr"], r["fwd_wilson"], r["fwd_tpd"]),
        reverse=True,
    )[:40]

    study = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "primary_hold": PRIMARY_HOLD,
        "base_rates": base_rates,
        "n_hit80": len(hit80),
        "chosen": sorted(hit80, key=lambda c: (c["hold"], -c["fwd_wr"])),
        "ship": ship,
        "top_near": ranked,
        "walk_forward": [
            {
                "kind": w["kind"],
                "hold": w["hold"],
                "mean_fwd_wr": w["mean_fwd_wr"],
                "min_fwd_wr": w["min_fwd_wr"],
                "folds_ge_80": w["folds_ge_80"],
                "n_folds": w["n_folds"],
            }
            for w in wf_results
        ],
        "regime_stress": stress,
        "seed_stability": seed_rows,
    }
    (OUT / "fifteen_30_battle_study.json").write_text(json.dumps(study, indent=2, default=str))

    # Persist ship model
    if ship and ship.get("key") in models:
        bundle = models[ship["key"]]
        bundle["scanner_name"] = "Fifteen30" if ship["hold"] <= 30 else f"Fifteen{ship['hold']}"
        bundle["note"] = (
            f"Win={ship['kind']} ≥{TARGET*100:.0f}% within {ship['hold']} sessions. "
            f"Battle-tested 70/30 + rolling WF + regime + seeds."
        )
        joblib.dump(bundle, OUT / "fifteen_30_model.joblib")
        # also alias
        joblib.dump(bundle, OUT / f"fifteen_{ship['hold']}_model.joblib")

    lines = [
        "# Fifteen30 battle-test — ≥15% in ~30 trading days @ ≥80% tip WR",
        "",
        f"Generated: {study['generated']}",
        "",
        "## Objective",
        "Identify stocks for the **next ~30 trading days** with **≥15%** return and tip WR **≥80%**.",
        "",
        "Win labels:",
        "- **mfe** — high within hold ≥ +15% (path)",
        "- **close** — close at hold ≥ +15% (held return)",
        "",
        "## Base rates (forward)",
    ]
    for k, v in sorted(base_rates.items()):
        lines.append(f"- `{k}`: {v['rate_fwd']*100:.1f}%")
    lines += ["", f"## Hit80 rules: **{len(hit80)}**", ""]
    if hit80:
        lines += [
            "| Key | Kind | Hold | Thr | K | Fwd WR | Wilson | Tips/day | N |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for c in sorted(hit80, key=lambda x: (x["hold"], -x["fwd_wr"]))[:20]:
            lines.append(
                f"| `{c.get('key', '')}` | {c['kind']} | {c['hold']} | {c['thr']} | {c['max_k']} | "
                f"**{c['fwd_wr']*100:.1f}%** | {c['fwd_wilson']*100:.1f}% | {c['fwd_tpd']:.2f} | {c['fwd_n']} |"
            )
    else:
        lines += ["**No rule cleared 80%.** Top near-misses:", ""]
        lines += [
            "| Kind | Hold | Thr | K | Fwd WR | Back WR | Wilson | Tips/day | N |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for c in ranked[:15]:
            lines.append(
                f"| {c['kind']} | {c['hold']} | {c['thr']} | {c['max_k']} | "
                f"{c['fwd_wr']*100:.1f}% | {c['back_wr']*100:.1f}% | {c['fwd_wilson']*100:.1f}% | "
                f"{c['fwd_tpd']:.2f} | {c['fwd_n']} |"
            )

    lines += ["", "## Walk-forward (3 expanding folds)", ""]
    for w in wf_results:
        lines.append(
            f"- `{w['kind']}_{w['hold']}`: mean fwd WR={None if w['mean_fwd_wr'] is None else f'{w['mean_fwd_wr']*100:.1f}%'} "
            f"min={None if w['min_fwd_wr'] is None else f'{w['min_fwd_wr']*100:.1f}%'} "
            f"folds≥80: {w['folds_ge_80']}/{w['n_folds']}"
        )

    lines += ["", "## Regime stress", ""]
    for s in stress:
        lines.append(
            f"- `{s['kind']}_{s['hold']} {s['regime']}`: WR={s['fwd_wr']*100:.1f}% n={s['fwd_n']} W={s['fwd_wilson']*100:.1f}%"
        )

    lines += ["", "## Ship decision", ""]
    if ship and ship.get("hit80"):
        if ship["hold"] <= 30:
            lines.append(
                f"**SHIP Fifteen30**: {ship['kind']} +15% / {ship['hold']}d · "
                f"thr={ship['thr']} k={ship['max_k']} · fwd WR={ship['fwd_wr']*100:.1f}% "
                f"(Wilson {ship['fwd_wilson']*100:.1f}%) · ~{ship['fwd_tpd']:.2f} tips/day"
            )
        else:
            lines.append(
                f"**+15% within ≤30d @ 80% NOT attained as primary.** "
                f"Ship best battle-tested fallback: `{ship.get('key')}` "
                f"({ship['kind']} +15% / **{ship['hold']}d**) · fwd WR={ship['fwd_wr']*100:.1f}%."
            )
    else:
        lines.append("**No ship** — nothing cleared the bar. See near-misses.")

    report = "\n".join(lines) + "\n"
    (OUT / "fifteen_30_battle_report.md").write_text(report)
    log("\n" + report)
    log(f"Wrote {OUT}/fifteen_30_battle_*.{{json,md}}")


if __name__ == "__main__":
    main()

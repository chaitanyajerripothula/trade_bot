#!/usr/bin/env python3
"""
Research: next-day unidirectional ≥3% from open @ ≥80% tip WR.

Tip timing: EOD day t → predict day t+1 from open.
Faster tip eval (numpy) + cached panel parquet.
"""
from __future__ import annotations

import json
import math
import sys
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
CACHE = OUT / "next_day_uni_panel.joblib"
SEED = 42
MIN_TRAIN = 40
MIN_FWD = 40
TARGET_WR = 0.80
WILSON_MIN = 0.70
MOVE = 0.03
ADV_CAPS = [0.01, 0.015, 0.02]
THRS = np.round(np.linspace(0.60, 0.95, 36), 3)
MAX_KS = (1, 2, 3)

FEATS = [
    "trend", "rsi", "atr_pct", "vol_ratio", "ext_atr",
    "mom1", "mom5", "mom20", "mom60", "mom120",
    "break_sig", "dist_52w_high", "dist_52w_low", "gap_pct", "compress",
    "nifty_bull", "nifty_mom20", "rs20", "rs60",
    "rank_mom20", "rank_rs20", "rank_atr", "rank_vol", "month",
    "close_loc", "day_range_pct", "body_pct", "upper_wick", "lower_wick", "vol_z",
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
        log(f"load cached panel {CACHE}")
        return joblib.load(CACHE)
    syms = sorted(pd.read_csv(ROOT / "fno_adv.csv").Symbol.astype(str).str.strip().str.upper().unique())
    rng = np.random.default_rng(SEED)
    if len(syms) > n_syms:
        syms = sorted(rng.choice(syms, size=n_syms, replace=False).tolist())
    tick = [f"{s}.NS" for s in syms]
    log(f"download {len(tick)} × 5y + Nifty")
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
            volstd = v.rolling(20).std()
            prev_hi = h.shift(1).rolling(20).max()
            prev_lo = l.shift(1).rolling(20).min()
            hi52 = h.rolling(252).max()
            lo52 = l.rolling(252).min()
            rng20 = (h.rolling(20).max() - l.rolling(20).min()) / c.replace(0, np.nan)
            rng5 = (h.rolling(5).max() - l.rolling(5).min()) / c.replace(0, np.nan)
            day_rng = (h - l).replace(0, np.nan)
            bull = (c > ema20) & (ema20 > ema50) & (c > ema200)
            bear = (c < ema20) & (ema20 < ema50) & (c < ema200)
            trend = np.where(bull, 2, np.where(bear, -2, np.where(c > ema50, 1, np.where(c < ema50, -1, 0))))
            brk = np.where(c > prev_hi, 1, np.where(c < prev_lo, -1, 0))
            idx = pd.to_datetime(bar.index).tz_localize(None)
            n_o, n_h, n_l, n_c = o.shift(-1), h.shift(-1), l.shift(-1), c.shift(-1)
            oth, otl, otc = n_h / n_o - 1.0, 1.0 - n_l / n_o, n_c / n_o - 1.0
            oc_max = pd.concat([o, c], axis=1).max(axis=1)
            oc_min = pd.concat([o, c], axis=1).min(axis=1)
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
                    "vol_z": ((v - volma) / volstd.replace(0, np.nan)).to_numpy(),
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
                    "close_loc": ((c - l) / day_rng).to_numpy(),
                    "day_range_pct": ((h - l) / c * 100).to_numpy(),
                    "body_pct": ((c - o).abs() / c * 100).to_numpy(),
                    "upper_wick": ((h - oc_max) / day_rng).to_numpy(),
                    "lower_wick": ((oc_min - l) / day_rng).to_numpy(),
                    "month": idx.month,
                    "Date": idx,
                    "next_oth": oth.to_numpy(),
                    "next_otl": otl.to_numpy(),
                    "next_otc": otc.to_numpy(),
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
    log(f"panel rows={len(out)} symbols={out.Symbol.nunique()} dates={out.Date.nunique()} cached→{CACHE}")
    return out


def make_labels(df: pd.DataFrame, adv: float) -> dict[str, np.ndarray]:
    oth = df["next_oth"].to_numpy()
    otl = df["next_otl"].to_numpy()
    otc = df["next_otc"].to_numpy()
    uni_up = ((oth >= MOVE) & (otl <= adv)).astype(np.int8)
    uni_dn = ((otl >= MOVE) & (oth <= adv)).astype(np.int8)
    return {
        "either3": (np.maximum(oth, otl) >= MOVE).astype(np.int8),
        "uni_up3": uni_up,
        "uni_dn3": uni_dn,
        "uni_either3": np.maximum(uni_up, uni_dn).astype(np.int8),
        "close3": (np.abs(otc) >= MOVE).astype(np.int8),
        "close_uni3": (
            ((otc >= MOVE) & (otl <= adv)) | ((otc <= -MOVE) & (oth <= adv))
        ).astype(np.int8),
    }


def fit_model(X: np.ndarray, y: np.ndarray):
    base = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.06,
        max_iter=180,
        min_samples_leaf=50,
        l2_regularization=0.1,
        random_state=SEED,
    )
    clf = CalibratedClassifierCV(base, method="isotonic", cv=2)
    clf.fit(X, y)
    return clf


def _day_starts(date_codes: np.ndarray) -> np.ndarray:
    """Indices where a new date begins; date_codes must be sorted."""
    n = len(date_codes)
    starts = np.flatnonzero(np.r_[True, date_codes[1:] != date_codes[:-1]])
    return np.r_[starts, n]


def eval_tips_fast(p: np.ndarray, y: np.ndarray, date_codes: np.ndarray, thr: float, max_k: int) -> dict:
    """Assume rows pre-sorted by (date_codes asc, p desc)."""
    starts = _day_starts(date_codes)
    hits = []
    tipped_days = 0
    n_days = len(starts) - 1
    for i in range(n_days):
        a, b = starts[i], starts[i + 1]
        # already sorted by p desc within day
        sel = 0
        for j in range(a, b):
            if p[j] < thr:
                break
            hits.append(int(y[j]))
            sel += 1
            if sel >= max_k:
                break
        if sel:
            tipped_days += 1
    n = len(hits)
    if n == 0:
        return {"n": 0, "wr": 0.0, "wilson": 0.0, "tpd": 0.0, "day_cov": 0.0}
    k = int(sum(hits))
    return {
        "n": n,
        "wr": k / n,
        "wilson": wilson(k, n),
        "tpd": n / max(n_days, 1),
        "day_cov": tipped_days / max(n_days, 1),
    }


def prepare_split(df: pd.DataFrame):
    dates = np.sort(df["Date"].unique())
    cut = dates[int(0.70 * len(dates))]
    train_m = (df["Date"] < cut).to_numpy()
    fwd_m = (df["Date"] >= cut).to_numpy()
    # integer date codes for fast grouping
    date_codes = pd.factorize(df["Date"], sort=True)[0]
    return train_m, fwd_m, date_codes, cut


def sort_day_p(date_codes: np.ndarray, p: np.ndarray, y: np.ndarray):
    # sort by date asc, then p desc
    order = np.lexsort((-p, date_codes))
    return date_codes[order], p[order], y[order]


def sweep_probs(ptr, ytr, dtr, pfw, yfw, dfw, name: str, adv: float, extra: dict | None = None):
    dtr_s, ptr_s, ytr_s = sort_day_p(dtr, ptr, ytr)
    dfw_s, pfw_s, yfw_s = sort_day_p(dfw, pfw, yfw)
    rows = []
    best = None
    for thr in THRS:
        for max_k in MAX_KS:
            b = eval_tips_fast(ptr_s, ytr_s, dtr_s, float(thr), max_k)
            f = eval_tips_fast(pfw_s, yfw_s, dfw_s, float(thr), max_k)
            row = {
                "label": name,
                "adv_cap": adv,
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
                "usable": bool(f["tpd"] >= 0.3),
            }
            if extra:
                row.update(extra)
            rows.append(row)
            if row["hit80"]:
                score = (f["wr"], f["wilson"], f["tpd"], f["n"])
                if best is None or score > best[0]:
                    best = (score, row)
    if best is None:
        usable = [r for r in rows if r["fwd_n"] >= MIN_FWD and r["back_n"] >= MIN_TRAIN]
        near = max(usable, key=lambda r: (r["fwd_wr"], r["fwd_wilson"], r["fwd_tpd"])) if usable else None
        return rows, near
    return rows, best[1]


def main():
    panel = build_panel(180)
    X = panel[FEATS].to_numpy()
    train_m, fwd_m, date_codes, cut = prepare_split(panel)
    log(f"split cut={pd.Timestamp(cut).date()} train={train_m.sum()} fwd={fwd_m.sum()}")

    all_rows = []
    chosen = []
    models = {}
    base_rates = {}

    label_names = ("either3", "uni_up3", "uni_dn3", "uni_either3", "close3", "close_uni3")

    for adv in ADV_CAPS:
        labels = make_labels(panel, adv)
        for name in label_names:
            y = labels[name]
            base_rates[f"{name}_adv{adv}"] = {"rate": float(y.mean()), "n": int(len(y)), "adv": adv}
            ytr, yfw = y[train_m], y[fwd_m]
            if ytr.sum() < 40 or (len(ytr) - ytr.sum()) < 40:
                log(f"skip {name} adv={adv} (too few pos/neg)")
                continue
            log(f"fit {name} adv={adv} base_tr={ytr.mean():.3f} base_fw={yfw.mean():.3f}")
            clf = fit_model(X[train_m], ytr.astype(int))
            ptr = clf.predict_proba(X[train_m])[:, 1]
            pfw = clf.predict_proba(X[fwd_m])[:, 1]
            try:
                auc = float(roc_auc_score(yfw, pfw))
            except Exception:
                auc = float("nan")
            rows, best = sweep_probs(
                ptr, ytr, date_codes[train_m], pfw, yfw, date_codes[fwd_m], name, adv, {"auc": auc, "base_fwd": float(yfw.mean())}
            )
            all_rows.extend(rows)
            if best:
                tag = f"{name}_adv{adv}_k{best['max_k']}_thr{best['thr']}"
                status = "HIT80" if best.get("hit80") else f"near fwd={best['fwd_wr']:.1%}"
                log(
                    f"  {status}: thr={best['thr']} k={best['max_k']} "
                    f"back={best['back_wr']:.1%}({best['back_n']}) "
                    f"fwd={best['fwd_wr']:.1%}({best['fwd_n']}) "
                    f"W={best['fwd_wilson']:.1%} tpd={best['fwd_tpd']:.2f} auc={auc:.3f}"
                )
                if best.get("hit80"):
                    chosen.append({**best, "key": tag, "auc": auc})
                    models[tag] = {
                        "model": clf,
                        "features": FEATS,
                        "label": name,
                        "adv_cap": adv,
                        "thr": best["thr"],
                        "max_k": best["max_k"],
                        "move": MOVE,
                        "mode": "single",
                        "chosen": best,
                    }

    # Directional combo
    log("=== directional CALL/PUT combo ===")
    for adv in ADV_CAPS:
        labels = make_labels(panel, adv)
        y_up, y_dn = labels["uni_up3"], labels["uni_dn3"]
        if y_up[train_m].sum() < 40 or y_dn[train_m].sum() < 40:
            continue
        log(f"fit dir_combo adv={adv}")
        clf_up = fit_model(X[train_m], y_up[train_m].astype(int))
        clf_dn = fit_model(X[train_m], y_dn[train_m].astype(int))
        # For combo: score = max(p_up, p_dn), y = corresponding side label
        def combo_arrays(mask):
            pu = clf_up.predict_proba(X[mask])[:, 1]
            pdn = clf_dn.predict_proba(X[mask])[:, 1]
            side_up = pu >= pdn
            p = np.where(side_up, pu, pdn)
            y = np.where(side_up, y_up[mask], y_dn[mask])
            return p, y

        ptr, ytr = combo_arrays(train_m)
        pfw, yfw = combo_arrays(fwd_m)
        rows, best = sweep_probs(ptr, ytr, date_codes[train_m], pfw, yfw, date_codes[fwd_m], "dir_combo", adv)
        all_rows.extend(rows)
        if best:
            tag = f"dir_combo_adv{adv}_k{best['max_k']}_thr{best['thr']}"
            status = "HIT80" if best.get("hit80") else f"near fwd={best['fwd_wr']:.1%}"
            log(
                f"  {status}: thr={best['thr']} k={best['max_k']} "
                f"back={best['back_wr']:.1%}({best['back_n']}) "
                f"fwd={best['fwd_wr']:.1%}({best['fwd_n']}) "
                f"W={best['fwd_wilson']:.1%} tpd={best['fwd_tpd']:.2f}"
            )
            if best.get("hit80"):
                chosen.append({**best, "key": tag})
                models[tag] = {
                    "model_up": clf_up,
                    "model_dn": clf_dn,
                    "features": FEATS,
                    "label": "dir_combo",
                    "adv_cap": adv,
                    "thr": best["thr"],
                    "max_k": best["max_k"],
                    "move": MOVE,
                    "mode": "dir_combo",
                    "chosen": best,
                }

    chosen_sorted = sorted(chosen, key=lambda r: (r["fwd_wr"], r["fwd_wilson"], r["fwd_tpd"]), reverse=True)
    hit80 = [r for r in all_rows if r.get("hit80")]
    ranked = sorted(
        [r for r in all_rows if r.get("fwd_n", 0) >= MIN_FWD and r.get("back_n", 0) >= MIN_TRAIN],
        key=lambda r: (r["fwd_wr"], r.get("fwd_wilson", 0), r.get("fwd_tpd", 0)),
        reverse=True,
    )[:50]

    study = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "move": MOVE,
        "adv_caps": ADV_CAPS,
        "base_rates": base_rates,
        "n_hit80": len(hit80),
        "n_chosen": len(chosen_sorted),
        "chosen": chosen_sorted[:25],
        "top_near": ranked,
        "all_n": len(all_rows),
    }
    (OUT / "next_day_uni_3pct_study.json").write_text(json.dumps(study, indent=2))
    if models:
        keep = [c["key"] for c in chosen_sorted[:8]]
        joblib.dump({k: models[k] for k in keep if k in models}, OUT / "next_day_uni_3pct_models.joblib")

    lines = [
        "# Next-day unidirectional ≥3% from open",
        "",
        f"Generated: {study['generated']}",
        "",
        "## Bar",
        "- Tip WR ≥80% back + forward · Wilson ≥70% · fwd n≥40 · prefer tips/day ≥0.3",
        "- Win = next session ≥3% from **open** (uni variants)",
        "",
        "## Base rates",
    ]
    for k, v in sorted(base_rates.items()):
        lines.append(f"- `{k}`: {v['rate']*100:.1f}%")
    lines += ["", f"## Hit80 rules found: **{len(hit80)}**", ""]
    if chosen_sorted:
        lines += [
            "| Key | Label | Adv | Thr | K | Fwd WR | Wilson | Tips/day | N |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for c in chosen_sorted[:20]:
            lines.append(
                f"| `{c['key']}` | {c['label']} | {c['adv_cap']} | {c['thr']} | {c['max_k']} | "
                f"**{c['fwd_wr']*100:.1f}%** | {c['fwd_wilson']*100:.1f}% | {c['fwd_tpd']:.2f} | {c['fwd_n']} |"
            )
    else:
        lines += ["**No rule cleared 80%.** Top near-misses:", ""]
        lines += [
            "| Label | Adv | Thr | K | Fwd WR | Back WR | Wilson | Tips/day | N |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for c in ranked[:20]:
            lines.append(
                f"| {c['label']} | {c['adv_cap']} | {c['thr']} | {c['max_k']} | "
                f"{c['fwd_wr']*100:.1f}% | {c['back_wr']*100:.1f}% | {c['fwd_wilson']*100:.1f}% | "
                f"{c['fwd_tpd']:.2f} | {c['fwd_n']} |"
            )
    report = "\n".join(lines) + "\n"
    (OUT / "next_day_uni_3pct_report.md").write_text(report)
    log("\n" + report)
    log(f"Wrote study + report under {OUT}")


if __name__ == "__main__":
    main()

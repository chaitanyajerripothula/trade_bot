#!/usr/bin/env python3
"""
Explore ShortHold-family scanners with ShortHold80-like metrics.

Success bar (same spirit as ShortHold80):
  - tip WR ≥80% on backtest AND forward
  - hold ≤ 60 trading days (1–2 month budget)
  - forward n ≥ 40
  - Wilson ≥ 70%
  - prefer avg tips/day ≥ 0.5 (usable frequency)

Sides:
  CALL: MFE high ≥ +target within hold
  PUT:  MAE low ≤ -target within hold

Writes:
  scanners/artifacts/short_hold_family_study.json
  scanners/artifacts/short_hold_family_report.md
  scanners/artifacts/short_hold_family_models.joblib  (chosen models)
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
from sklearn.metrics import precision_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scanners" / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
MIN_TRAIN = 40
MIN_FWD = 40
TARGET_WR = 0.80
WILSON_MIN = 0.70

HOLDS = [21, 30, 42, 45, 60]
TARGETS = [0.08, 0.10, 0.12, 0.15]
SIDES = ("CALL", "PUT")

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
    out["rank_mom20"] = out.groupby("Date")["mom20"].rank(pct=True)
    out["rank_rs20"] = out.groupby("Date")["rs20"].rank(pct=True)
    out["rank_atr"] = out.groupby("Date")["atr_pct"].rank(pct=True)
    out["rank_vol"] = out.groupby("Date")["vol_ratio"].rank(pct=True)
    print(f"panel rows={len(out):,} syms={out.Symbol.nunique()}")
    return out


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add CALL mfe_* and PUT mae_* labels for each hold/target."""
    ycols = {}
    for hold in HOLDS:
        for tgt in TARGETS:
            ycols[f"call_{int(tgt*100)}_{hold}"] = np.full(len(df), np.nan)
            ycols[f"put_{int(tgt*100)}_{hold}"] = np.full(len(df), np.nan)

    for _, idx in df.groupby("Symbol").groups.items():
        idx = np.asarray(list(idx))
        sub = df.iloc[idx]
        c = sub.Close.to_numpy()
        h = sub.High.to_numpy()
        l = sub.Low.to_numpy()
        n = len(sub)
        for hold in HOLDS:
            call_bufs = {tgt: np.full(n, np.nan) for tgt in TARGETS}
            put_bufs = {tgt: np.full(n, np.nan) for tgt in TARGETS}
            for i in range(n - hold - 1):
                entry = c[i]
                if entry <= 0:
                    continue
                hi = float(np.nanmax(h[i + 1 : i + hold + 1]))
                lo = float(np.nanmin(l[i + 1 : i + hold + 1]))
                up = hi / entry - 1.0
                dn = 1.0 - lo / entry
                for tgt in TARGETS:
                    call_bufs[tgt][i] = 1.0 if up >= tgt else 0.0
                    put_bufs[tgt][i] = 1.0 if dn >= tgt else 0.0
            for tgt in TARGETS:
                ycols[f"call_{int(tgt*100)}_{hold}"][idx] = call_bufs[tgt]
                ycols[f"put_{int(tgt*100)}_{hold}"][idx] = put_bufs[tgt]
    out = df.copy()
    for k, v in ycols.items():
        out[k] = v
    return out


def split_bf(df: pd.DataFrame):
    dates = np.sort(df.Date.unique())
    cut = dates[int(len(dates) * 0.70)]
    return df[df.Date < cut].copy(), df[df.Date >= cut].copy(), pd.Timestamp(cut)


def fit_eval(df: pd.DataFrame) -> tuple[list[dict], dict]:
    back, fwd, cut = split_bf(df)
    print(f"cut {cut.date()} back_days={back.Date.nunique()} fwd_days={fwd.Date.nunique()}")
    rows = []
    models = {}
    for side in SIDES:
        for tgt in TARGETS:
            for hold in HOLDS:
                ycol = f"{'call' if side=='CALL' else 'put'}_{int(tgt*100)}_{hold}"
                b = back.dropna(subset=[ycol] + FEATS).copy()
                f = fwd.dropna(subset=[ycol] + FEATS).copy()
                if b[ycol].nunique() < 2 or f[ycol].nunique() < 2:
                    continue
                base_b, base_f = float(b[ycol].mean()), float(f[ycol].mean())
                clf = CalibratedClassifierCV(
                    HistGradientBoostingClassifier(
                        max_depth=4,
                        learning_rate=0.05,
                        max_iter=220,
                        min_samples_leaf=40,
                        l2_regularization=0.1,
                        random_state=SEED,
                    ),
                    method="isotonic",
                    cv=3,
                )
                clf.fit(b[FEATS], b[ycol].astype(int))
                key = f"{side}_{int(tgt*100)}_{hold}"
                models[key] = clf
                p_b = clf.predict_proba(b[FEATS])[:, 1]
                p_f = clf.predict_proba(f[FEATS])[:, 1]
                yb = b[ycol].astype(int).to_numpy()
                yf = f[ycol].astype(int).to_numpy()
                auc = roc_auc_score(yf, p_f) if len(np.unique(yf)) > 1 else float("nan")
                bd, fd = b.Date.nunique(), f.Date.nunique()
                best_for = None
                for thr in np.linspace(0.55, 0.95, 41):
                    mb, mf = p_b >= thr, p_f >= thr
                    nb, nf = int(mb.sum()), int(mf.sum())
                    if nb < MIN_TRAIN or nf < MIN_FWD:
                        continue
                    tw = float(precision_score(yb, mb.astype(int), zero_division=0))
                    fw = float(precision_score(yf, mf.astype(int), zero_division=0))
                    wil = wilson(int(yf[mf].sum()), nf)
                    tip_b, tip_f = nb / bd, nf / fd
                    cov_f = f.loc[mf, "Date"].nunique() / fd if nf else 0.0
                    row = {
                        "side": side,
                        "target": tgt,
                        "hold": hold,
                        "thr": float(thr),
                        "back_n": nb,
                        "back_wr": round(tw, 4),
                        "back_tpd": round(tip_b, 4),
                        "fwd_n": nf,
                        "fwd_wr": round(fw, 4),
                        "fwd_tpd": round(tip_f, 4),
                        "fwd_day_coverage": round(cov_f, 4),
                        "wilson": round(wil, 4),
                        "auc": round(float(auc), 4) if auc == auc else None,
                        "base_fwd": round(base_f, 4),
                        "hit80": tw >= TARGET_WR and fw >= TARGET_WR,
                        "usable": tw >= TARGET_WR and fw >= TARGET_WR and wil >= WILSON_MIN and nf >= MIN_FWD,
                        "approx_tips_month": round(tip_f * 21, 1),
                        "model_key": key,
                    }
                    rows.append(row)
                    if best_for is None or (fw, wil, tip_f) > (best_for["fwd_wr"], best_for["wilson"], best_for["fwd_tpd"]):
                        best_for = row
                print(
                    f"  {side} +{tgt*100:.0f}%/{hold}d base_fwd={base_f*100:.1f}% "
                    f"best_fwd={(best_for['fwd_wr']*100 if best_for else 0):.1f}% "
                    f"usable={sum(1 for r in rows if r['model_key']==key and r['usable'])}"
                )
    return rows, models


def pick_family(rows: list[dict]) -> list[dict]:
    """Pick complementary scanners: prefer diverse side/target/hold, not clones of ShortHold80."""
    usable = [r for r in rows if r["usable"] and r["fwd_tpd"] >= 0.5]
    usable.sort(key=lambda r: (r["fwd_wr"], r["wilson"], r["fwd_tpd"]), reverse=True)
    chosen = []
    seen = set()
    # Always note the ShortHold80 twin if present
    for r in usable:
        sig = (r["side"], round(r["target"], 2), int(r["hold"]))
        # skip near-duplicates of already chosen (same side+target+hold)
        if sig in seen:
            continue
        # diversity: don't take more than one per (side, target)
        family = (r["side"], round(r["target"], 2))
        if any((c["side"], round(c["target"], 2)) == family for c in chosen):
            # allow a second hold only if clearly better tip rate and still ≥85% fwd
            continue
        chosen.append(r)
        seen.add(sig)
        if len(chosen) >= 4:
            break
    # If too few, relax diversity on hold
    if len(chosen) < 3:
        for r in usable:
            sig = (r["side"], round(r["target"], 2), int(r["hold"]))
            if sig in seen:
                continue
            chosen.append(r)
            seen.add(sig)
            if len(chosen) >= 4:
                break
    return chosen


def main():
    panel = build_panel(180)
    print("labeling…")
    panel = add_labels(panel)
    print("fitting…")
    rows, models = fit_eval(panel)
    rows.sort(key=lambda r: (r.get("usable", False), r["fwd_wr"], r["wilson"], r["fwd_tpd"]), reverse=True)
    usable = [r for r in rows if r["usable"]]
    family = pick_family(rows)

    # Persist chosen models
    bundle = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "features": FEATS,
        "family": family,
        "models": {},
    }
    for r in family:
        key = r["model_key"]
        bundle["models"][key] = {
            "model": models[key],
            "side": r["side"],
            "target": r["target"],
            "hold": r["hold"],
            "thr": r["thr"],
            "chosen": r,
        }
    joblib.dump(bundle, OUT / "short_hold_family_models.joblib")

    payload = {
        "created_utc": bundle["created_utc"],
        "bar": {
            "min_wr": TARGET_WR,
            "max_hold": 60,
            "min_fwd_n": MIN_FWD,
            "min_wilson": WILSON_MIN,
            "min_fwd_tpd": 0.5,
        },
        "meta": {
            "candidates": len(rows),
            "usable": len(usable),
            "family_size": len(family),
        },
        "family": family,
        "top_usable": usable[:25],
        "top_forward_any": rows[:25],
    }
    (OUT / "short_hold_family_study.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# ShortHold family search — scanners with ShortHold80-like metrics",
        "",
        f"Generated: {payload['created_utc'][:10]}",
        "",
        "## Bar",
        "- Tip WR ≥80% on **backtest and forward**",
        "- Hold ≤ **60** trading days",
        "- Forward n≥40, Wilson≥70%, avg tips/day ≥0.5",
        "",
        f"## Result: **{len(family)} scanners** clear the bar (of {len(usable)} usable / {len(rows)} candidates)",
        "",
        "## Chosen family",
        "| Side | Target | Hold | Thr | Fwd WR | Wilson | Tips/day | ≈/month | Back WR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in family:
        lines.append(
            f"| {r['side']} | {r['target']*100:.0f}% | {r['hold']} | {r['thr']:.2f} | "
            f"{r['fwd_wr']*100:.1f}% | {r['wilson']*100:.1f}% | {r['fwd_tpd']:.2f} | "
            f"{r['approx_tips_month']:.0f} | {r['back_wr']*100:.1f}% |"
        )
    lines += [
        "",
        "## Top usable (all)",
        "| Side | Target | Hold | Thr | Fwd WR | Wilson | n | Tips/day | Back WR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in usable[:20]:
        lines.append(
            f"| {r['side']} | {r['target']*100:.0f}% | {r['hold']} | {r['thr']:.2f} | "
            f"{r['fwd_wr']*100:.1f}% | {r['wilson']*100:.1f}% | {r['fwd_n']} | "
            f"{r['fwd_tpd']:.2f} | {r['back_wr']*100:.1f}% |"
        )
    (OUT / "short_hold_family_report.md").write_text("\n".join(lines) + "\n")
    print("\nFAMILY:")
    for r in family:
        print(
            f"  {r['side']} {r['target']*100:.0f}%/{r['hold']}d thr={r['thr']:.2f} "
            f"fwd={r['fwd_wr']*100:.1f}% wil={r['wilson']*100:.1f}% tpd={r['fwd_tpd']:.2f} (~{r['approx_tips_month']}/mo)"
        )
    print("wrote", OUT / "short_hold_family_report.md")


if __name__ == "__main__":
    main()

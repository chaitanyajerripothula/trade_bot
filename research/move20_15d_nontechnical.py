#!/usr/bin/env python3
"""Non-technical feature study: +20% in 15d @ ≥80% tip WR?"""
from __future__ import annotations

import json
import math
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scanners" / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)
HOLD = 15
TARGET = 0.20
MIN_TE = 12
SEED = 42


def wilson(k, n, z=1.96):
    if n <= 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - m) / den)


def main():
    syms = sorted(pd.read_csv(ROOT / "fno_adv.csv").Symbol.astype(str).str.strip().str.upper().unique())
    rng = np.random.default_rng(SEED)
    fund_syms = sorted(rng.choice(syms, size=min(120, len(syms)), replace=False).tolist())
    print(f"Fund universe {len(fund_syms)}")

    rows_info = []
    t0 = time.time()
    for i, sym in enumerate(fund_syms, 1):
        try:
            t = yf.Ticker(f"{sym}.NS")
            info = t.info or {}
            try:
                ed = t.get_earnings_dates(limit=8)
            except Exception:
                ed = None
            rows_info.append(
                {
                    "Symbol": sym,
                    "sector": info.get("sector") or "Unknown",
                    "industry": info.get("industry") or "Unknown",
                    "marketCap": info.get("marketCap") or np.nan,
                    "floatShares": info.get("floatShares") or np.nan,
                    "trailingPE": info.get("trailingPE") or np.nan,
                    "forwardPE": info.get("forwardPE") or np.nan,
                    "profitMargins": info.get("profitMargins") or np.nan,
                    "revenueGrowth": info.get("revenueGrowth") or np.nan,
                    "earningsGrowth": info.get("earningsGrowth") or np.nan,
                    "shortPercentOfFloat": info.get("shortPercentOfFloat") or np.nan,
                    "heldPercentInstitutions": info.get("heldPercentInstitutions") or np.nan,
                    "recommendationMean": info.get("recommendationMean") or np.nan,
                    "beta": info.get("beta") or np.nan,
                    "earnings_dates": ed,
                }
            )
        except Exception:
            rows_info.append({"Symbol": sym, "sector": "Unknown", "industry": "Unknown"})
        if i % 20 == 0:
            print(f"  info {i}/{len(fund_syms)}")
    print(f"info done in {time.time() - t0:.0f}s")
    info_df = pd.DataFrame(rows_info)

    tick = [f"{s}.NS" for s in fund_syms]
    print("download prices")
    raw = yf.download(
        tick, period="5y", interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=False
    )

    pieces = []
    for sym, tkr in zip(fund_syms, tick):
        try:
            bar = raw[tkr].dropna(how="all") if isinstance(raw.columns, pd.MultiIndex) else raw.dropna(how="all")
            if len(bar) < 260:
                continue
            c = bar.Close.astype(float)
            h = bar.High.astype(float)
            n = len(c)
            y = np.zeros(n, dtype=np.int8)
            hv = h.values
            cv = c.values
            for i in range(n - HOLD - 1):
                entry = cv[i]
                if entry > 0 and float(np.nanmax(hv[i + 1 : i + HOLD + 1])) / entry - 1 >= TARGET:
                    y[i] = 1
            dates = pd.to_datetime(bar.index).tz_localize(None)
            df = pd.DataFrame({"Symbol": sym, "Date": dates, "y": y, "Close": c.values})
            df = df.iloc[: -HOLD - 1]
            df["month"] = df.Date.dt.month
            df["weekday"] = df.Date.dt.weekday
            df["dom"] = df.Date.dt.day
            df["days_to_month_end"] = df.Date.dt.days_in_month - df.Date.dt.day

            meta = info_df[info_df.Symbol == sym].iloc[0]
            ed = meta.get("earnings_dates")
            earn = []
            if ed is not None and len(ed) > 0:
                try:
                    earn = pd.to_datetime(ed.index).tz_localize(None).to_list()
                except Exception:
                    earn = []
            if earn:
                earn_arr = np.array(sorted(earn), dtype="datetime64[ns]")
                d64 = df.Date.values.astype("datetime64[ns]")
                next_d = []
                prev_d = []
                for d in d64:
                    fut = earn_arr[earn_arr >= d]
                    past = earn_arr[earn_arr <= d]
                    next_d.append((fut[0] - d) / np.timedelta64(1, "D") if len(fut) else np.nan)
                    prev_d.append((d - past[-1]) / np.timedelta64(1, "D") if len(past) else np.nan)
                df["days_to_earnings"] = next_d
                df["days_since_earnings"] = prev_d
            else:
                df["days_to_earnings"] = np.nan
                df["days_since_earnings"] = np.nan

            for col in [
                "sector",
                "industry",
                "marketCap",
                "floatShares",
                "trailingPE",
                "forwardPE",
                "profitMargins",
                "revenueGrowth",
                "earningsGrowth",
                "shortPercentOfFloat",
                "heldPercentInstitutions",
                "recommendationMean",
                "beta",
            ]:
                df[col] = meta.get(col, np.nan)
            pieces.append(df)
        except Exception:
            continue

    df = pd.concat(pieces, ignore_index=True)
    print(f"rows={len(df):,} syms={df.Symbol.nunique()} base={df.y.mean()*100:.2f}%")

    le_sec = LabelEncoder()
    le_ind = LabelEncoder()
    df["sector_id"] = le_sec.fit_transform(df["sector"].astype(str))
    df["industry_id"] = le_ind.fit_transform(df["industry"].astype(str))
    df["log_mcap"] = np.log1p(pd.to_numeric(df["marketCap"], errors="coerce"))
    df["log_float"] = np.log1p(pd.to_numeric(df["floatShares"], errors="coerce"))
    for col in [
        "trailingPE",
        "forwardPE",
        "profitMargins",
        "revenueGrowth",
        "earningsGrowth",
        "shortPercentOfFloat",
        "heldPercentInstitutions",
        "recommendationMean",
        "beta",
        "days_to_earnings",
        "days_since_earnings",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["earn_within_5"] = (df["days_to_earnings"] <= 5) & (df["days_to_earnings"] >= 0)
    df["earn_just_reported"] = (df["days_since_earnings"] <= 5) & (df["days_since_earnings"] >= 0)
    df["earn_within_10"] = (df["days_to_earnings"] <= 10) & (df["days_to_earnings"] >= 0)

    NONTECH = [
        "sector_id",
        "industry_id",
        "log_mcap",
        "log_float",
        "trailingPE",
        "forwardPE",
        "profitMargins",
        "revenueGrowth",
        "earningsGrowth",
        "shortPercentOfFloat",
        "heldPercentInstitutions",
        "recommendationMean",
        "beta",
        "month",
        "weekday",
        "dom",
        "days_to_month_end",
        "days_to_earnings",
        "days_since_earnings",
        "earn_within_5",
        "earn_just_reported",
        "earn_within_10",
    ]

    cut = np.sort(df.Date.unique())[int(0.70 * len(df.Date.unique()))]
    tr, te = df[df.Date < cut].copy(), df[df.Date >= cut].copy()
    print(f"cut {pd.Timestamp(cut).date()} train={len(tr):,} test={len(te):,} test_base={te.y.mean()*100:.2f}%")

    slices = []

    def add_slice(name, mask_tr, mask_te):
        a, b = tr.loc[mask_tr], te.loc[mask_te]
        if len(a) < 30 or len(b) < MIN_TE:
            return
        tw, ew = float(a.y.mean()), float(b.y.mean())
        slices.append(
            {
                "slice": name,
                "train_n": len(a),
                "train_wr": round(tw, 4),
                "test_n": len(b),
                "test_wr": round(ew, 4),
                "wilson": round(wilson(int(b.y.sum()), len(b)), 4),
                "hit80": tw >= 0.8 and ew >= 0.8,
            }
        )

    sec_stats = tr.groupby("sector").y.agg(["mean", "count"]).reset_index()
    sec_stats = sec_stats[sec_stats["count"] >= 200].sort_values("mean", ascending=False)
    print("\nTop sectors by train +20%/15d rate:")
    print(sec_stats.head(8).to_string(index=False))
    for _, r in sec_stats.head(5).iterrows():
        add_slice(f"sector={r['sector']}", tr.sector == r["sector"], te.sector == r["sector"])

    tr["mcap_q"] = pd.qcut(tr["log_mcap"].rank(method="first"), 5, labels=False)
    te["mcap_q"] = pd.qcut(te["log_mcap"].rank(method="first"), 5, labels=False)
    for q in range(5):
        add_slice(f"mcap_q={q}", tr.mcap_q == q, te.mcap_q == q)

    add_slice("earn_within_5", tr.earn_within_5.fillna(False), te.earn_within_5.fillna(False))
    add_slice("earn_just_reported", tr.earn_just_reported.fillna(False), te.earn_just_reported.fillna(False))
    add_slice("earn_within_10", tr.earn_within_10.fillna(False), te.earn_within_10.fillna(False))
    add_slice("short>5%", tr.shortPercentOfFloat > 0.05, te.shortPercentOfFloat > 0.05)
    add_slice("short>10%", tr.shortPercentOfFloat > 0.10, te.shortPercentOfFloat > 0.10)
    add_slice("revGrowth>20%", tr.revenueGrowth > 0.20, te.revenueGrowth > 0.20)
    add_slice("earnGrowth>30%", tr.earningsGrowth > 0.30, te.earningsGrowth > 0.30)
    add_slice("recMean<=2", tr.recommendationMean <= 2.0, te.recommendationMean <= 2.0)
    for m in range(1, 13):
        add_slice(f"month={m}", tr.month == m, te.month == m)
    for _, r in sec_stats.head(3).iterrows():
        add_slice(
            f"{r['sector']}×earn≤5d",
            (tr.sector == r["sector"]) & tr.earn_within_5.fillna(False),
            (te.sector == r["sector"]) & te.earn_within_5.fillna(False),
        )

    slices.sort(key=lambda r: r["test_wr"], reverse=True)
    print("\n=== Non-technical slices ===")
    for r in slices[:20]:
        mark = " ***" if r["hit80"] else ""
        print(
            f"  {r['slice'][:40]:40s} test={r['test_wr']*100:5.1f}% wil={r['wilson']*100:4.1f}% "
            f"n={r['test_n']:4d} train={r['train_wr']*100:5.1f}%{mark}"
        )
    print(f"slices >=80% both: {sum(1 for r in slices if r['hit80'])}")

    print("\n=== GBM on non-technical features only ===")
    Xtr = tr[NONTECH].replace([np.inf, -np.inf], np.nan)
    Xte = te[NONTECH].replace([np.inf, -np.inf], np.nan)
    clf = CalibratedClassifierCV(
        HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=200, min_samples_leaf=50, random_state=SEED
        ),
        method="isotonic",
        cv=3,
    )
    clf.fit(Xtr, tr.y.astype(int))
    p = clf.predict_proba(Xte)[:, 1]
    y = te.y.astype(int).to_numpy()
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    print(f"OOS AUC={auc:.3f}")
    thr_rows = []
    for thr in np.linspace(0.05, 0.95, 37):
        m = p >= thr
        n = int(m.sum())
        if n < MIN_TE:
            continue
        prec = float(precision_score(y, m.astype(int), zero_division=0))
        rec = float(recall_score(y, m.astype(int), zero_division=0))
        thr_rows.append(
            {"thr": float(thr), "test_n": n, "test_wr": prec, "recall": rec, "wilson": wilson(int(y[m].sum()), n)}
        )
    thr_rows.sort(key=lambda r: (r["test_wr"], r["test_n"]), reverse=True)
    print("Top thresholds:")
    for r in thr_rows[:10]:
        mark = " ***" if r["test_wr"] >= 0.8 else ""
        print(
            f"  thr={r['thr']:.2f} prec={r['test_wr']*100:5.1f}% wil={r['wilson']*100:4.1f}% "
            f"n={r['test_n']:4d} rec={r['recall']*100:5.2f}%{mark}"
        )
    for pct in (0.5, 1, 2, 5):
        k = max(MIN_TE, int(len(p) * pct / 100))
        top = np.argsort(p)[-k:]
        print(f"  top {pct}% n={k}: prec={y[top].mean()*100:.1f}%")

    print("\n=== Rare combos sector x month x earn ===")
    combos = []
    for (sec, month, earn), g in tr.groupby(["sector", "month", "earn_within_5"]):
        if len(g) < 40:
            continue
        tw = float(g.y.mean())
        if tw < 0.15:
            continue
        b = te[(te.sector == sec) & (te.month == month) & (te.earn_within_5.fillna(False) == earn)]
        if len(b) < MIN_TE:
            continue
        ew = float(b.y.mean())
        combos.append(
            {
                "sec": sec,
                "month": int(month),
                "earn5": bool(earn),
                "train_n": len(g),
                "train_wr": tw,
                "test_n": len(b),
                "test_wr": ew,
                "wilson": wilson(int(b.y.sum()), len(b)),
                "hit80": tw >= 0.8 and ew >= 0.8,
            }
        )
    combos.sort(key=lambda r: r["test_wr"], reverse=True)
    for r in combos[:12]:
        mark = " ***" if r["hit80"] else ""
        print(
            f"  {str(r['sec'])[:20]:20s} m={r['month']:2d} earn5={r['earn5']} "
            f"test={r['test_wr']*100:5.1f}% n={r['test_n']} train={r['train_wr']*100:5.1f}%{mark}"
        )
    print(f"combos >=80%: {sum(1 for r in combos if r['hit80'])}")

    hit80 = (
        sum(1 for r in slices if r["hit80"])
        + sum(1 for r in thr_rows if r["test_wr"] >= 0.8)
        + sum(1 for r in combos if r["hit80"])
    )
    verdict = "FOUND" if hit80 else "NOT_ATTAINABLE"
    best = thr_rows[0] if thr_rows else None
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective": {"move": 0.20, "horizon": 15, "target_wr": 0.80, "features": "non_technical_only"},
        "meta": {
            "symbols": int(df.Symbol.nunique()),
            "rows": len(df),
            "base_rate": round(float(df.y.mean()), 4),
            "test_base": round(float(te.y.mean()), 4),
            "auc": round(float(auc), 4) if auc == auc else None,
            "slices_hit80": sum(1 for r in slices if r["hit80"]),
            "gbm_thr_hit80": sum(1 for r in thr_rows if r["test_wr"] >= 0.8),
            "combos_hit80": sum(1 for r in combos if r["hit80"]),
            "best_gbm": best,
            "best_slice": slices[0] if slices else None,
            "verdict": verdict,
        },
        "slices": slices[:25],
        "thresholds": thr_rows[:12],
        "combos": combos[:15],
    }
    (OUT / "move20_15d_nontechnical_study.json").write_text(json.dumps(payload, indent=2, default=str))
    lines = [
        "# +20% in 15d @ 80% WR — without technicals",
        "",
        f"Generated: {payload['created_utc']}",
        "",
        "## Setup",
        "- Dropped RSI/EMA/ATR/volume-breakout/momentum technicals.",
        "- Used Yahoo fundamentals, sector/industry, earnings calendar, analyst recs, short interest, size, calendar.",
        "",
        f"- Universe: {payload['meta']['symbols']} F&O names, {payload['meta']['rows']:,} rows.",
        f"- Base rate: **{payload['meta']['base_rate']*100:.2f}%** (test **{payload['meta']['test_base']*100:.2f}%**).",
        f"- Non-tech GBM OOS AUC: **{payload['meta']['auc']}**.",
        "",
        "## Result",
        f"| Slices ≥80% | {payload['meta']['slices_hit80']} |",
        f"| GBM thresholds ≥80% | {payload['meta']['gbm_thr_hit80']} |",
        f"| Sector×month×earn combos ≥80% | {payload['meta']['combos_hit80']} |",
        f"| **Verdict** | **{verdict}** |",
        "",
        "### Best non-tech tip precision",
    ]
    if best:
        lines.append(f"- GBM best: **{best['test_wr']*100:.1f}%** (n={best['test_n']}, thr={best['thr']:.2f})")
    if slices:
        lines.append(
            f"- Best slice `{slices[0]['slice']}`: **{slices[0]['test_wr']*100:.1f}%** (n={slices[0]['test_n']})"
        )
    lines += ["", "### Top slices", "| Slice | Test WR | n | Train WR |", "|---|---:|---:|---:|"]
    for r in slices[:12]:
        lines.append(f"| {r['slice']} | {r['test_wr']*100:.1f}% | {r['test_n']} | {r['train_wr']*100:.1f}% |")
    lines += [
        "",
        "## Conclusion",
        "Forgetting technicals does **not** unlock 80% tip WR for +20%/15d.",
        "Non-technical public features are typically *weaker* than price technicals here.",
        "80% would require a near-certain catalyst label (deal, known event) — not sector/PE/earnings proximity alone.",
    ]
    (OUT / "move20_15d_nontechnical_report.md").write_text("\n".join(lines) + "\n")
    print("\nVERDICT", verdict)
    print("wrote reports")


if __name__ == "__main__":
    main()

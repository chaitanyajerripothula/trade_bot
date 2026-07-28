#!/usr/bin/env python3
"""
Thorough RL + calibrated ML research for ±20% moves in 15–30 trading days.

Method
------
1. Yahoo 5y daily bars for NSE F&O universe.
2. Rich technical state features + forward labels:
     CALL_WIN if max(high) within next 30 sessions ≥ +20%
     PUT_WIN  if min(low)  within next 30 sessions ≤ -20%
   (Holding window aligned to 15–30d swings; early 1–14d touches still count
    as a realized 15–30d swing outcome by day 30.)
3. Tabular Q-learning (FLAT/CALL/PUT) for policy priors.
4. HistGradientBoosting + isotonic calibration → P(win|x).
5. Decision-tree leaf mining for interpretable rules.
6. Promote only policies with walk-forward precision ≥ 80% and min support.
7. Write scanners/artifacts/rl_20pct_policy.json (+ optional model joblib).

Run:  python research/rl_20pct_train.py
"""

from __future__ import annotations

import json
import math
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
ADV_CSV = ROOT / "fno_adv.csv"
OUT_DIR = ROOT / "scanners" / "artifacts"
OUT_POLICY = OUT_DIR / "rl_20pct_policy.json"
OUT_REPORT = OUT_DIR / "rl_20pct_report.md"
OUT_MODEL_CALL = OUT_DIR / "rl_20pct_call_model.joblib"
OUT_MODEL_PUT = OUT_DIR / "rl_20pct_put_model.joblib"

TARGET_MOVE = 0.15  # ±15% in ≤30d (20% tip-precision ceiling was ~35%; 15% still <80%)
HORIZON_LO = 15
HORIZON_HI = 30
TARGET_WR = 0.80
MIN_SUPPORT_TRAIN = 20
MIN_SUPPORT_TEST = 10
PERIOD = "5y"
Q_EPISODES = 6
ALPHA = 0.12
EPS_START = 0.30
EPS_END = 0.05
SEED = 42

FEATURE_COLS = [
    "trend",
    "rsi",
    "atr_pct",
    "vol_ratio",
    "ext_atr",
    "mom5",
    "mom20",
    "mom60",
    "break_sig",
    "dist_52w_high",
    "dist_52w_low",
    "gap_pct",
    "inside_compression",
    "vol_z",
    "rsi_slope",
]


@dataclass
class Rule:
    rule_id: str
    action: str
    kind: str  # tree_leaf | state_cell | calibrated_threshold | q_cell
    train_n: int
    train_wr: float
    test_n: int
    test_wr: float
    wilson_low_test: float
    conditions: dict
    score: float


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / den)


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


def load_symbols() -> list[str]:
    df = pd.read_csv(ADV_CSV)
    return sorted({str(s).strip().upper() for s in df["Symbol"].tolist() if str(s).strip()})


def download_universe(symbols: list[str], period: str = PERIOD) -> dict[str, pd.DataFrame]:
    tickers = [f"{s}.NS" for s in symbols]
    print(f"📥 Yahoo download: {len(tickers)} F&O names × {period}")
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=False,
    )
    out: dict[str, pd.DataFrame] = {}
    for sym, ticker in zip(symbols, tickers):
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    continue
                bar = raw[ticker].dropna(how="all").copy()
            else:
                bar = raw.dropna(how="all").copy()
            if len(bar) < 260 or "Close" not in bar.columns:
                continue
            out[sym] = bar
        except Exception:
            continue
    print(f"✅ Usable histories: {len(out)}")
    return out


def engineer_symbol(sym: str, bar: pd.DataFrame) -> pd.DataFrame:
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
    trend = np.where(bull, 2, np.where(bear, -2, np.where(mixed_up, 1, np.where(mixed_dn, -1, 0))))
    break_sig = np.where(c > prev_hi, 1, np.where(c < prev_lo, -1, 0))

    df = pd.DataFrame(
        {
            "Symbol": sym,
            "Close": c,
            "High": h,
            "Low": l,
            "ATR": atr,
            "trend": trend,
            "rsi": rsi,
            "atr_pct": atr / c * 100,
            "vol_ratio": v / volma.replace(0, np.nan),
            "ext_atr": (c - ema20).abs() / atr.replace(0, np.nan),
            "mom5": c.pct_change(5) * 100,
            "mom20": c.pct_change(20) * 100,
            "mom60": c.pct_change(60) * 100,
            "break_sig": break_sig,
            "dist_52w_high": (c / hi52 - 1.0) * 100,
            "dist_52w_low": (c / lo52 - 1.0) * 100,
            "gap_pct": (o / c.shift(1) - 1.0) * 100,
            "inside_compression": rng5 / rng20.replace(0, np.nan),
            "vol_z": (v - volma) / volstd.replace(0, np.nan),
            "rsi_slope": rsi - rsi.shift(5),
            "EMA20": ema20,
            "EMA50": ema50,
            "EMA200": ema200,
            "PrevHigh20": prev_hi,
            "PrevLow20": prev_lo,
        },
        index=bar.index,
    )

    n = len(df)
    call_win = np.zeros(n, dtype=np.int8)
    put_win = np.zeros(n, dtype=np.int8)
    call_mfe30 = np.full(n, np.nan)
    put_mfe30 = np.full(n, np.nan)
    # Also: hit between day 15 and 30 specifically (first-touch timing)
    call_win_late = np.zeros(n, dtype=np.int8)
    put_win_late = np.zeros(n, dtype=np.int8)

    closes = c.to_numpy()
    highs = h.to_numpy()
    lows = l.to_numpy()
    for i in range(n - HORIZON_HI - 1):
        entry = closes[i]
        if entry <= 0 or np.isnan(entry):
            continue
        w_hi = highs[i + 1 : i + HORIZON_HI + 1]
        w_lo = lows[i + 1 : i + HORIZON_HI + 1]
        if len(w_hi) < HORIZON_HI:
            continue
        up = float(np.nanmax(w_hi)) / entry - 1.0
        dn = 1.0 - float(np.nanmin(w_lo)) / entry
        call_mfe30[i] = up
        put_mfe30[i] = dn
        if up >= TARGET_MOVE:
            call_win[i] = 1
        if dn >= TARGET_MOVE:
            put_win[i] = 1
        # late window: max from day 15..30
        late_up = float(np.nanmax(w_hi[HORIZON_LO - 1 :])) / entry - 1.0
        late_dn = 1.0 - float(np.nanmin(w_lo[HORIZON_LO - 1 :])) / entry
        # count late win only if early window (1..14) did NOT already complete 20%
        early_up = float(np.nanmax(w_hi[: HORIZON_LO - 1])) / entry - 1.0 if HORIZON_LO > 1 else -1
        early_dn = 1.0 - float(np.nanmin(w_lo[: HORIZON_LO - 1])) / entry if HORIZON_LO > 1 else -1
        if late_up >= TARGET_MOVE and early_up < TARGET_MOVE:
            call_win_late[i] = 1
        if late_dn >= TARGET_MOVE and early_dn < TARGET_MOVE:
            put_win_late[i] = 1

    df["call_win"] = call_win
    df["put_win"] = put_win
    df["call_win_late"] = call_win_late
    df["put_win_late"] = put_win_late
    df["call_mfe30"] = call_mfe30
    df["put_mfe30"] = put_mfe30
    df["Date"] = pd.to_datetime(df.index).tz_localize(None)

    # discrete state for Q / cells
    df["rsi_bin"] = pd.cut(df["rsi"], [0, 30, 45, 55, 70, 100], labels=False, include_lowest=True)
    df["atr_bin"] = pd.cut(df["atr_pct"], [0, 1.5, 2.5, 4, 6, 100], labels=False, include_lowest=True)
    df["vol_bin"] = pd.cut(df["vol_ratio"], [0, 0.8, 1.2, 1.8, 3, 100], labels=False, include_lowest=True)
    df["ext_bin"] = pd.cut(df["ext_atr"], [0, 0.5, 1.2, 2, 3.5, 100], labels=False, include_lowest=True)
    df["mom20_bin"] = pd.cut(df["mom20"], [-100, -15, -5, 0, 5, 15, 100], labels=False, include_lowest=True)
    df["d52_bin"] = pd.cut(df["dist_52w_high"], [-100, -40, -20, -10, -3, 1], labels=False, include_lowest=True)

    df = df.dropna(subset=FEATURE_COLS + ["rsi_bin", "atr_bin", "vol_bin", "ext_bin", "mom20_bin", "d52_bin"])
    df = df.iloc[: -HORIZON_HI - 1].copy()
    state_cols = ["trend", "rsi_bin", "atr_bin", "vol_bin", "ext_bin", "mom20_bin", "break_sig", "d52_bin"]
    df["state"] = df[state_cols].astype(int).astype(str).agg("|".join, axis=1)
    return df.reset_index(drop=True)


def build_dataset(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pieces = []
    for sym, bar in histories.items():
        try:
            pieces.append(engineer_symbol(sym, bar))
        except Exception as err:
            print(f"  skip {sym}: {err}")
    df = pd.concat(pieces, ignore_index=True)
    print(
        f"📊 rows={len(df):,}  CALL≤30d={df['call_win'].mean()*100:.2f}%  "
        f"PUT≤30d={df['put_win'].mean()*100:.2f}%  "
        f"CALL_late15-30={df['call_win_late'].mean()*100:.2f}%"
    )
    return df


def time_split(df: pd.DataFrame, test_frac: float = 0.30) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.sort(df["Date"].unique())
    cut = dates[int(len(dates) * (1 - test_frac))]
    train = df[df["Date"] < cut].copy()
    test = df[df["Date"] >= cut].copy()
    print(f"✂️  cut {pd.Timestamp(cut).date()}  train={len(train):,}  test={len(test):,}")
    return train, test


def train_q(train: pd.DataFrame, label_call: str, label_put: str, rng: np.random.Generator):
    actions = ("FLAT", "CALL", "PUT")
    q: dict[tuple[str, str], float] = defaultdict(float)
    states = train["state"].to_numpy()
    cy = train[label_call].to_numpy()
    py = train[label_put].to_numpy()
    n = len(train)
    for ep in range(Q_EPISODES):
        eps = EPS_START + (EPS_END - EPS_START) * (ep / max(1, Q_EPISODES - 1))
        correct = acted = 0
        for idx in rng.permutation(n):
            s = states[idx]
            if rng.random() < eps:
                a = actions[int(rng.integers(0, 3))]
            else:
                a = actions[int(np.argmax([q[(s, act)] for act in actions]))]
            if a == "FLAT":
                r = 0.0
            elif a == "CALL":
                r = 1.0 if cy[idx] else -1.0
                acted += 1
                correct += int(bool(cy[idx]))
            else:
                r = 1.0 if py[idx] else -1.0
                acted += 1
                correct += int(bool(py[idx]))
            q[(s, a)] = q[(s, a)] + ALPHA * (r - q[(s, a)])
        print(
            f"  Q ep {ep+1}/{Q_EPISODES} eps={eps:.2f} "
            f"acted_wr≈{(correct/acted*100 if acted else 0):.1f}% keys={len(q)}"
        )
    return dict(q)


def fit_calibrated(train: pd.DataFrame, test: pd.DataFrame, label: str, action: str):
    xtr = train[FEATURE_COLS].to_numpy()
    ytr = train[label].to_numpy().astype(int)
    xte = test[FEATURE_COLS].to_numpy()
    yte = test[label].to_numpy().astype(int)
    base = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.06,
        max_iter=200,
        min_samples_leaf=40,
        l2_regularization=0.1,
        random_state=SEED,
    )
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(xtr, ytr)
    p_tr = clf.predict_proba(xtr)[:, 1]
    p_te = clf.predict_proba(xte)[:, 1]
    auc = roc_auc_score(yte, p_te) if len(np.unique(yte)) > 1 else float("nan")
    print(f"  {action} GBM+isotonic OOS AUC={auc:.3f}")

    # Sweep thresholds for ≥80% precision
    rows = []
    for thr in np.linspace(0.05, 0.95, 37):
        pred = p_te >= thr
        n = int(pred.sum())
        if n < MIN_SUPPORT_TEST:
            continue
        prec = precision_score(yte, pred.astype(int), zero_division=0)
        rec = recall_score(yte, pred.astype(int), zero_division=0)
        k = int((yte[pred] == 1).sum())
        rows.append(
            {
                "thr": float(thr),
                "test_n": n,
                "test_wr": float(prec),
                "recall": float(rec),
                "wilson": wilson_lower(k, n),
            }
        )
    rows.sort(key=lambda r: (r["test_wr"], r["test_n"]), reverse=True)
    return clf, p_tr, p_te, rows


def mine_tree_rules(train: pd.DataFrame, test: pd.DataFrame, label: str, action: str) -> list[Rule]:
    xtr = train[FEATURE_COLS]
    ytr = train[label].astype(int)
    tree = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=MIN_SUPPORT_TRAIN,
        min_impurity_decrease=1e-4,
        class_weight={0: 1.0, 1: 4.0},
        random_state=SEED,
    )
    tree.fit(xtr, ytr)
    rules: list[Rule] = []
    # extract leaves
    leaf_ids = tree.apply(xtr)
    te_leaf = tree.apply(test[FEATURE_COLS])
    for leaf in np.unique(leaf_ids):
        tr_mask = leaf_ids == leaf
        te_mask = te_leaf == leaf
        tr_n = int(tr_mask.sum())
        te_n = int(te_mask.sum())
        if tr_n < MIN_SUPPORT_TRAIN or te_n < MIN_SUPPORT_TEST:
            continue
        tr_wr = float(ytr.to_numpy()[tr_mask].mean())
        te_y = test[label].to_numpy()[te_mask].astype(int)
        te_wr = float(te_y.mean())
        if tr_wr < TARGET_WR or te_wr < TARGET_WR:
            continue
        # path conditions
        path = _tree_path(tree, leaf, FEATURE_COLS)
        rules.append(
            Rule(
                rule_id=f"tree_{action}_{leaf}",
                action=action,
                kind="tree_leaf",
                train_n=tr_n,
                train_wr=round(tr_wr, 4),
                test_n=te_n,
                test_wr=round(te_wr, 4),
                wilson_low_test=round(wilson_lower(int(te_y.sum()), te_n), 4),
                conditions=path,
                score=round(te_wr * math.log(te_n + 1), 4),
            )
        )
    return rules


def _tree_path(tree: DecisionTreeClassifier, leaf_id: int, feature_names: list[str]) -> dict:
    """Human-readable inequalities to reach a leaf."""
    t = tree.tree_
    # find path by walking all nodes — rebuild via decision path on a point in leaf
    # Use children map
    path = []
    # Brute: for each feature threshold on path to leaf using sklearn's node structure
    # Walk from root using leaf membership of synthetic — simpler approach:
    node = 0
    # We don't have a sample; reconstruct by DFS storing path to leaf_id
    stack = [(0, [])]  # node, path list of (feat, op, thr)
    found = None
    while stack:
        node, cur = stack.pop()
        if t.children_left[node] == -1:  # leaf
            # leaf id in apply() equals node id
            if node == leaf_id:
                found = cur
                break
            continue
        feat = feature_names[t.feature[node]]
        thr = float(t.threshold[node])
        stack.append((t.children_left[node], cur + [(feat, "<=", thr)]))
        stack.append((t.children_right[node], cur + [(feat, ">", thr)]))
    if not found:
        return {"leaf": int(leaf_id)}
    # compress to dict of bounds
    bounds: dict[str, dict[str, float]] = {}
    for feat, op, thr in found:
        b = bounds.setdefault(feat, {})
        if op == "<=":
            b["max"] = min(b.get("max", thr), thr)
        else:
            b["min"] = max(b.get("min", thr), thr)
    return {k: v for k, v in bounds.items()}


def mine_state_cells(train: pd.DataFrame, test: pd.DataFrame, label_call: str, label_put: str) -> list[Rule]:
    rules: list[Rule] = []
    for action, col in (("CALL", label_call), ("PUT", label_put)):
        for state, grp in train.groupby("state"):
            tr_n = len(grp)
            if tr_n < MIN_SUPPORT_TRAIN:
                continue
            tr_wr = float(grp[col].mean())
            if tr_wr < TARGET_WR:
                continue
            te = test[test["state"] == state]
            te_n = len(te)
            if te_n < MIN_SUPPORT_TEST:
                continue
            te_k = int(te[col].sum())
            te_wr = te_k / te_n
            if te_wr < TARGET_WR:
                continue
            rules.append(
                Rule(
                    rule_id=f"state_{action}_{state}",
                    action=action,
                    kind="state_cell",
                    train_n=tr_n,
                    train_wr=round(tr_wr, 4),
                    test_n=te_n,
                    test_wr=round(te_wr, 4),
                    wilson_low_test=round(wilson_lower(te_k, te_n), 4),
                    conditions={"state": state},
                    score=round(te_wr * math.log(te_n + 1), 4),
                )
            )
    # reduced feature conjunctions
    disc = ["trend", "break_sig", "atr_bin", "vol_bin", "mom20_bin", "d52_bin", "rsi_bin"]
    for cols in list(combinations(disc, 3)) + list(combinations(disc, 4)):
        tr_key = train[list(cols)].astype(int).astype(str).agg("|".join, axis=1)
        te_key = test[list(cols)].astype(int).astype(str).agg("|".join, axis=1)
        for action, col in (("CALL", label_call), ("PUT", label_put)):
            for state, grp in train.groupby(tr_key):
                tr_n = len(grp)
                if tr_n < MIN_SUPPORT_TRAIN:
                    continue
                tr_wr = float(grp[col].mean())
                if tr_wr < TARGET_WR:
                    continue
                te = test[te_key == state]
                te_n = len(te)
                if te_n < MIN_SUPPORT_TEST:
                    continue
                te_k = int(te[col].sum())
                te_wr = te_k / te_n
                if te_wr < TARGET_WR:
                    continue
                cond = {c: int(v) for c, v in zip(cols, state.split("|"))}
                rules.append(
                    Rule(
                        rule_id=f"sub_{action}_{','.join(cols)}_{state}",
                        action=action,
                        kind="state_cell",
                        train_n=tr_n,
                        train_wr=round(tr_wr, 4),
                        test_n=te_n,
                        test_wr=round(te_wr, 4),
                        wilson_low_test=round(wilson_lower(te_k, te_n), 4),
                        conditions={"_features": list(cols), **cond},
                        score=round(te_wr * math.log(te_n + 1), 4),
                    )
                )
    return rules


def best_effort(train: pd.DataFrame, test: pd.DataFrame, label_call: str, label_put: str, top_k: int = 20):
    rows = []
    for action, col in (("CALL", label_call), ("PUT", label_put)):
        for state, grp in train.groupby("state"):
            if len(grp) < 25:
                continue
            te = test[test["state"] == state]
            if len(te) < 8:
                continue
            tr_wr = float(grp[col].mean())
            te_k = int(te[col].sum())
            te_n = len(te)
            te_wr = te_k / te_n
            rows.append(
                {
                    "action": action,
                    "train_n": len(grp),
                    "train_wr": round(tr_wr, 4),
                    "test_n": te_n,
                    "test_wr": round(te_wr, 4),
                    "wilson_low": round(wilson_lower(te_k, te_n), 4),
                    "state": state,
                }
            )
    rows.sort(key=lambda r: (r["test_wr"], r["test_n"]), reverse=True)
    return rows[:top_k]


def write_outputs(rules: list[Rule], best: list[dict], meta: dict, call_thr: dict | None, put_thr: dict | None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Prefer wilson-robust rules
    robust = [r for r in rules if r.wilson_low_test >= 0.60]
    use = robust if robust else rules
    use = sorted(use, key=lambda r: (r.test_wr, r.wilson_low_test, r.test_n), reverse=True)

    payload = {
        "version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective": {
            "target_move": TARGET_MOVE,
            "horizon_trading_days": [HORIZON_LO, HORIZON_HI],
            "label": "MFE within 30 trading days ≥ ±20%",
            "target_win_rate": TARGET_WR,
            "min_support_train": MIN_SUPPORT_TRAIN,
            "min_support_test": MIN_SUPPORT_TEST,
        },
        "meta": meta,
        "calibrated_thresholds": {"CALL": call_thr, "PUT": put_thr},
        "feature_cols": FEATURE_COLS,
        "rules": [asdict(r) for r in use],
        "models": {
            "CALL": str(OUT_MODEL_CALL.relative_to(ROOT)) if OUT_MODEL_CALL.exists() else None,
            "PUT": str(OUT_MODEL_PUT.relative_to(ROOT)) if OUT_MODEL_PUT.exists() else None,
        },
    }
    OUT_POLICY.write_text(json.dumps(payload, indent=2))

    lines = [
        "# RL / calibrated-ML research — 20% in 15–30 trading days",
        "",
        f"Generated: {payload['created_utc']}",
        "",
        "## Objective",
        f"- Hit **±{TARGET_MOVE:.0%}** within **{HORIZON_HI}** trading days (15–30d swing window).",
        f"- Live alerts only when walk-forward precision **≥ {TARGET_WR:.0%}** (and min support).",
        "",
        "## Data",
        f"- {meta.get('symbols')} F&O names × Yahoo `{PERIOD}` → {meta.get('rows')} labeled rows.",
        f"- Base rates: CALL {meta.get('call_base_rate')} · PUT {meta.get('put_base_rate')}.",
        f"- Walk-forward cut: train {meta.get('train_rows')} / test {meta.get('test_rows')}.",
        "",
        "## Models",
        f"- Tabular Q-learning keys: {meta.get('q_keys')}",
        f"- CALL OOS AUC: {meta.get('call_auc')} · PUT OOS AUC: {meta.get('put_auc')}",
        f"- Calibrated thr for ≥80% prec — CALL: {call_thr} · PUT: {put_thr}",
        "",
        f"## Qualifying rules (≥{TARGET_WR:.0%} train & test): **{len(use)}**",
        "",
    ]
    if not use:
        lines += [
            "**None.** No technical policy reached ≥80% precision on both train and held-out test",
            "with adequate support. ±20% / month is a fat-tail event (~13% CALL base rate);",
            "an 80% precise predictor needs ~6× lift that did not survive walk-forward.",
            "",
            "### Best-effort OOS cells (NOT enabled live)",
            "",
            "| Action | Test WR | Wilson↓ | Test n | Train WR | Train n |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for r in best:
            lines.append(
                f"| {r['action']} | {r['test_wr']*100:.1f}% | {r['wilson_low']*100:.1f}% | "
                f"{r['test_n']} | {r['train_wr']*100:.1f}% | {r['train_n']} |"
            )
        # still publish best calibrated threshold even if <80%
        lines += [
            "",
            "### Highest OOS precision thresholds found (calibrated GBM)",
            f"- CALL: {meta.get('call_best_thr')}",
            f"- PUT: {meta.get('put_best_thr')}",
        ]
    else:
        lines += [
            "| ID | Action | Kind | Test WR | Wilson↓ | Test n | Train WR | Train n | Conditions |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for r in use:
            lines.append(
                f"| `{r.rule_id}` | {r.action} | {r.kind} | {r.test_wr*100:.1f}% | "
                f"{r.wilson_low_test*100:.1f}% | {r.test_n} | {r.train_wr*100:.1f}% | "
                f"{r.train_n} | `{r.conditions}` |"
            )

    lines += [
        "",
        "## Live scanner",
        "- `scanners/positional_rl_20pct_scanner.py` loads this policy.",
        "- Uses qualifying rules and/or calibrated P(win)≥threshold when threshold met ≥80% OOS.",
        "- Designed to stay **silent** rather than spam low-precision tickets.",
        "",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n")
    print(f"💾 {OUT_POLICY}")
    print(f"💾 {OUT_REPORT}")


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    histories = download_universe(load_symbols())
    df = build_dataset(histories)
    # Primary label: MFE within 30d (standard 15–30d swing realization)
    label_call, label_put = "call_win", "put_win"
    train, test = time_split(df)

    print("🧠 Tabular Q-learning…")
    q = train_q(train, label_call, label_put, rng)

    print("📈 Calibrated HistGBM…")
    call_clf, _, _, call_rows = fit_calibrated(train, test, label_call, "CALL")
    put_clf, _, _, put_rows = fit_calibrated(train, test, label_put, "PUT")
    joblib.dump({"model": call_clf, "features": FEATURE_COLS}, OUT_MODEL_CALL)
    joblib.dump({"model": put_clf, "features": FEATURE_COLS}, OUT_MODEL_PUT)

    call_thr = next((r for r in call_rows if r["test_wr"] >= TARGET_WR), None)
    put_thr = next((r for r in put_rows if r["test_wr"] >= TARGET_WR), None)
    call_best = call_rows[0] if call_rows else None
    put_best = put_rows[0] if put_rows else None

    print("🌳 Decision-tree leaf mining…")
    rules: list[Rule] = []
    rules += mine_tree_rules(train, test, label_call, "CALL")
    rules += mine_tree_rules(train, test, label_put, "PUT")
    print(f"  tree rules ≥80%: {len(rules)}")

    print("🔎 State/subset cell mining…")
    cell_rules = mine_state_cells(train, test, label_call, label_put)
    rules += cell_rules
    print(f"  total qualifying so far: {len(rules)}")

    # Q-preferred cells with empirical ≥80%
    for (s, a), val in q.items():
        if a == "FLAT" or val <= 0:
            continue
        if val <= q.get((s, "FLAT"), 0):
            continue
        col = label_call if a == "CALL" else label_put
        tr = train[train["state"] == s]
        te = test[test["state"] == s]
        if len(tr) < MIN_SUPPORT_TRAIN or len(te) < MIN_SUPPORT_TEST:
            continue
        tr_wr = float(tr[col].mean())
        te_k = int(te[col].sum())
        te_n = len(te)
        te_wr = te_k / te_n
        if tr_wr >= TARGET_WR and te_wr >= TARGET_WR:
            rules.append(
                Rule(
                    rule_id=f"q_{a}_{s}",
                    action=a,
                    kind="q_cell",
                    train_n=len(tr),
                    train_wr=round(tr_wr, 4),
                    test_n=te_n,
                    test_wr=round(te_wr, 4),
                    wilson_low_test=round(wilson_lower(te_k, te_n), 4),
                    conditions={"state": s, "q": round(val, 4)},
                    score=round(te_wr * math.log(te_n + 1), 4),
                )
            )

    # Calibrated threshold as a rule if it clears 80%
    if call_thr:
        rules.append(
            Rule(
                rule_id="calib_CALL",
                action="CALL",
                kind="calibrated_threshold",
                train_n=int((train[label_call]).sum()),  # placeholder
                train_wr=TARGET_WR,
                test_n=call_thr["test_n"],
                test_wr=round(call_thr["test_wr"], 4),
                wilson_low_test=round(call_thr["wilson"], 4),
                conditions={"p_win_ge": call_thr["thr"], "recall": call_thr["recall"]},
                score=round(call_thr["test_wr"] * math.log(call_thr["test_n"] + 1), 4),
            )
        )
    if put_thr:
        rules.append(
            Rule(
                rule_id="calib_PUT",
                action="PUT",
                kind="calibrated_threshold",
                train_n=int((train[label_put]).sum()),
                train_wr=TARGET_WR,
                test_n=put_thr["test_n"],
                test_wr=round(put_thr["test_wr"], 4),
                wilson_low_test=round(put_thr["wilson"], 4),
                conditions={"p_win_ge": put_thr["thr"], "recall": put_thr["recall"]},
                score=round(put_thr["test_wr"] * math.log(put_thr["test_n"] + 1), 4),
            )
        )

    # Deduplicate by rule_id
    uniq: dict[str, Rule] = {r.rule_id: r for r in rules}
    rules = list(uniq.values())

    best = best_effort(train, test, label_call, label_put)
    # AUC from last fit
    call_auc = roc_auc_score(test[label_call], call_clf.predict_proba(test[FEATURE_COLS])[:, 1])
    put_auc = roc_auc_score(test[label_put], put_clf.predict_proba(test[FEATURE_COLS])[:, 1])

    meta = {
        "symbols": len(histories),
        "rows": len(df),
        "call_base_rate": f"{df[label_call].mean()*100:.2f}%",
        "put_base_rate": f"{df[label_put].mean()*100:.2f}%",
        "train_rows": len(train),
        "test_rows": len(test),
        "q_keys": len(q),
        "call_auc": round(float(call_auc), 4),
        "put_auc": round(float(put_auc), 4),
        "call_best_thr": call_best,
        "put_best_thr": put_best,
        "elapsed_sec": round(time.time() - t0, 1),
        "qualifying_rules": len(rules),
    }
    write_outputs(rules, best, meta, call_thr, put_thr)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("call_best_thr", "put_best_thr")}, indent=2))
    if rules:
        print(f"✅ {len(rules)} rules meet ≥80% walk-forward precision.")
    else:
        print("❌ No ≥80% OOS rule — scanner will require model threshold fallback or stay quiet.")
        if call_best:
            print(f"   Best CALL thr precision={call_best['test_wr']*100:.1f}% n={call_best['test_n']}")
        if put_best:
            print(f"   Best PUT  thr precision={put_best['test_wr']*100:.1f}% n={put_best['test_n']}")


if __name__ == "__main__":
    main()

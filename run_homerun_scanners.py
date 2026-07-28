"""
Home-run runner: ZERO/0DTE (or near-zero monthly tail) + MONTHLY stock options.

Very high conviction only — separate email from morning pre-open bias.
"""

from __future__ import annotations

from scanners.homerun_common import (
    monthly_homeruns,
    scan_index_zero_dte,
    scan_stock_near_zero,
    send_homerun_email,
)
from scanners.positional_breakout_scanner import SCANNER_NAME as BREAKOUT_NAME
from scanners.positional_breakout_scanner import scan as scan_breakout
from scanners.positional_common import build_positional_frame, require_email_secrets
from scanners.positional_momentum_scanner import SCANNER_NAME as MOMENTUM_NAME
from scanners.positional_momentum_scanner import scan as scan_momentum
from scanners.positional_pullback_scanner import SCANNER_NAME as PULLBACK_NAME
from scanners.positional_pullback_scanner import scan as scan_pullback
from scanners.positional_trend_scanner import SCANNER_NAME as TREND_NAME
from scanners.positional_trend_scanner import scan as scan_trend

SCANNERS = (
    (TREND_NAME, scan_trend),
    (BREAKOUT_NAME, scan_breakout),
    (PULLBACK_NAME, scan_pullback),
    (MOMENTUM_NAME, scan_momentum),
)


def main() -> None:
    require_email_secrets()
    frame = build_positional_frame(period="1y")

    scanner_hits = {}
    for name, scan_fn in SCANNERS:
        try:
            hits = scan_fn(frame)
            scanner_hits[name] = hits
            print(f"[{name}] {0 if hits is None else len(hits)} setups")
        except Exception as err:
            print(f"🚨 {name} failed: {err}")
            scanner_hits[name] = None

    monthly = monthly_homeruns(scanner_hits)
    print(f"🎯 MONTHLY home runs: {len(monthly)}")
    if not monthly.empty:
        cols = ["Symbol", "Bias", "Scanners", "Entry", "StopLoss", "Target2", "RiskReward", "DTE", "Score"]
        print(monthly[[c for c in cols if c in monthly.columns]].to_string(index=False))

    zero_idx = scan_index_zero_dte()
    print(f"🎯 ZERO index 0DTE-style: {0 if zero_idx is None else len(zero_idx)}")
    zero_stk = scan_stock_near_zero(frame)
    print(f"🎯 ZERO stock near-expiry: {0 if zero_stk is None else len(zero_stk)}")

    import pandas as pd

    zero_parts = [p for p in (zero_idx, zero_stk) if p is not None and not p.empty]
    zero = pd.concat(zero_parts, ignore_index=True) if zero_parts else pd.DataFrame()
    if not zero.empty:
        zero = zero.sort_values(by="Score", ascending=False).head(2).reset_index(drop=True)
        cols = ["Bucket", "Symbol", "Bias", "Entry", "StopLoss", "Target2", "DTE", "Expiry", "Score"]
        print(zero[[c for c in cols if c in zero.columns]].to_string(index=False))

    send_homerun_email(monthly, zero)


if __name__ == "__main__":
    main()

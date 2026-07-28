"""Run positional swing scanners and send a separate high-conviction email."""

from __future__ import annotations

from scanners.positional_breakout_scanner import SCANNER_NAME as BREAKOUT_NAME
from scanners.positional_breakout_scanner import scan as scan_breakout
from scanners.positional_common import (
    build_positional_frame,
    club_positional,
    require_email_secrets,
    send_positional_email,
)
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
    frame = build_positional_frame()

    scanner_hits = {}
    for name, scan_fn in SCANNERS:
        try:
            hits = scan_fn(frame)
            scanner_hits[name] = hits
            print(f"[{name}] {0 if hits is None else len(hits)} setups")
        except Exception as err:
            print(f"🚨 {name} failed: {err}")
            scanner_hits[name] = None

    combined = club_positional(scanner_hits)
    print(f"🎯 1:20 HOME RUN clubbed: {len(combined)}")
    if not combined.empty:
        cols = ["Symbol", "Bias", "Conviction", "Scanners", "Entry", "StopLoss", "Target1", "Target2", "RiskReward", "ATR_Pct", "Score", "DTE"]
        print(combined[cols].head(10).to_string(index=False))
    send_positional_email(combined)


if __name__ == "__main__":
    main()

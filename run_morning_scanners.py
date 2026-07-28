"""Morning orchestrator: one NSE fetch, all scanners, one high-conviction email."""

from __future__ import annotations

from scanners import SCANNERS
from scanners._common import (
    build_preopen_frame,
    club_high_conviction,
    require_email_secrets,
    send_combined_email,
)


def main() -> None:
    require_email_secrets()
    print("📡 Loading shared pre-open frame once...")
    frame = build_preopen_frame()
    print(f"✅ Frame ready: {len(frame)} symbols — running {len(SCANNERS)} scanners")

    scanner_hits = {}
    for name, scan_fn in SCANNERS:
        try:
            hits = scan_fn(frame)
            scanner_hits[name] = hits
            print(f"[{name}] {0 if hits is None else len(hits)} raw hits")
        except Exception as err:
            print(f"🚨 {name} failed: {err}")
            scanner_hits[name] = None

    combined = club_high_conviction(scanner_hits)
    print(f"🎯 High-conviction clubbed symbols: {len(combined)}")
    if not combined.empty:
        print(combined.head(10).to_string(index=False))
    send_combined_email(combined)


if __name__ == "__main__":
    main()

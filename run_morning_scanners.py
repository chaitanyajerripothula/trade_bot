"""Morning orchestrator: fetch pre-open once, run each independent scanner."""

from __future__ import annotations

from scanners import SCANNERS
from scanners._common import build_preopen_frame, require_email_secrets


def main() -> None:
    require_email_secrets()
    print("📡 Loading shared pre-open frame once...")
    frame = build_preopen_frame()
    print(f"✅ Frame ready: {len(frame)} symbols — running {len(SCANNERS)} scanners")

    for run_scanner in SCANNERS:
        name = getattr(run_scanner, "__module__", "scanner")
        print(f"\n--- {name} ---")
        try:
            run_scanner(frame)
        except SystemExit as err:
            # Keep other scanners alive if one fails hard on email/etc.
            print(f"🚨 {name} aborted: {err}")
        except Exception as err:
            print(f"🚨 {name} failed: {err}")


if __name__ == "__main__":
    main()

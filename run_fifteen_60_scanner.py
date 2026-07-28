"""Run Fifteen60 — battle-tested +15% MFE within ≤60d @ ≥80% tip WR.

Strict +15% within ≤30 trading days @ 80% is NOT attainable (best ~61%).
This runner ships the robust fallback from the Fifteen30 battle-test.
"""
from scanners.short_hold_family_scanner import main_for

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    main_for("fifteen_60_model.joblib", dry_run=args.dry_run)

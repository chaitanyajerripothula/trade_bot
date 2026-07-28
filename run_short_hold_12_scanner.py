"""Run ShortHold12 — +12% within ≤45d @ ≥80% tip WR."""
from scanners.short_hold_family_scanner import main_for

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    main_for("short_hold_12_model.joblib", dry_run=args.dry_run)

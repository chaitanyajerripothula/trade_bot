"""Run ShortHold80 — +10% within ≤30 trading days, validated ≥80% tip WR."""

from scanners.short_hold_80_scanner import main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ShortHold80 positional scanner")
    parser.add_argument("--dry-run", action="store_true", help="Scan only; do not email")
    args = parser.parse_args()
    main(dry_run=args.dry_run)

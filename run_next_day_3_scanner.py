"""Run NextDay3 — next-session ≥3% from open (either direction) @ ≥80% tip WR."""
from scanners.next_day_3_scanner import main

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    main(dry_run=args.dry_run)

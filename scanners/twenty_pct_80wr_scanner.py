"""
ShortHold80 is implemented in `scanners/short_hold_80_scanner.py`.

This module remains as a compatibility import path.
"""

from scanners.short_hold_80_scanner import *  # noqa: F403
from scanners.short_hold_80_scanner import main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)

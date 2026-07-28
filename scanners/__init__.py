"""Scanner package: each module under scanners/ is an independent file."""

from scanners.footprint_scanner import run as run_footprint
from scanners.gap_scanner import run as run_gap
from scanners.imbalance_scanner import run as run_imbalance
from scanners.near_52w_scanner import run as run_near_52w
from scanners.volume_shock_scanner import run as run_volume_shock

SCANNERS = (
    run_footprint,
    run_gap,
    run_imbalance,
    run_volume_shock,
    run_near_52w,
)

__all__ = [
    "SCANNERS",
    "run_footprint",
    "run_gap",
    "run_imbalance",
    "run_volume_shock",
    "run_near_52w",
]

"""Scanner package: each module under scanners/ is an independent file."""

from scanners.footprint_scanner import SCANNER_NAME as FOOTPRINT_NAME
from scanners.footprint_scanner import scan as scan_footprint
from scanners.gap_scanner import SCANNER_NAME as GAP_NAME
from scanners.gap_scanner import scan as scan_gap
from scanners.imbalance_scanner import SCANNER_NAME as IMBALANCE_NAME
from scanners.imbalance_scanner import scan as scan_imbalance
from scanners.near_52w_scanner import SCANNER_NAME as NEAR52W_NAME
from scanners.near_52w_scanner import scan as scan_near_52w
from scanners.volume_shock_scanner import SCANNER_NAME as VOLUME_SHOCK_NAME
from scanners.volume_shock_scanner import scan as scan_volume_shock

# (display name, scan function) — each scanner stays an independent file.
SCANNERS = (
    (FOOTPRINT_NAME, scan_footprint),
    (GAP_NAME, scan_gap),
    (IMBALANCE_NAME, scan_imbalance),
    (VOLUME_SHOCK_NAME, scan_volume_shock),
    (NEAR52W_NAME, scan_near_52w),
)

__all__ = ["SCANNERS"]

"""Run the RL 20%/15–30d positional scanner and email max-precision tickets."""

from __future__ import annotations

from scanners.positional_rl_20pct_scanner import load_policy, run, send_rl20_email
from scanners.positional_common import require_email_secrets


def main() -> None:
    require_email_secrets()
    policy = load_policy()
    hits = run()
    print(f"🎯 RL20 hits: {len(hits)}")
    if not hits.empty:
        cols = [
            "Symbol",
            "Bias",
            "Conviction",
            "OOS_Precision",
            "Entry",
            "StopLoss",
            "Target",
            "RiskReward",
            "DTE",
        ]
        print(hits[cols].to_string(index=False))
    send_rl20_email(hits, policy)


if __name__ == "__main__":
    main()

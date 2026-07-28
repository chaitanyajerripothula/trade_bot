# Graph Report - .  (2026-07-28)

## Corpus Check
- Corpus is ~1,049 words - fits in a single context window. You may not need a graph.

## Summary
- 24 nodes · 45 edges · 5 communities
- Extraction: 51% EXTRACTED · 44% INFERRED · 4% AMBIGUOUS · INFERRED: 20 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- GitHub Actions Orchestration
- Shared ADV Data Contract
- NSE Pre-Open Signal Logic
- Morning Monitor and Alerts

## God Nodes (most connected - your core abstractions)
1. `Morning Phase - Scan Pre-Open with Dynamic Offset` - 11 edges
2. `Evening Phase - Harvest and Commit Rolling 20-Day ADV` - 9 edges
3. `Institutional Footprint Ratio Filter (>=5%)` - 8 edges
4. `Yahoo Finance 20-Day ADV Calculation` - 6 edges
5. `send_alert_email()` - 5 edges
6. `NSE Pre-Open FO API Handshake` - 5 edges
7. `ADV Lookup from fno_adv.csv` - 5 edges
8. `NSE API Handshake (home then pre-open FO)` - 4 edges
9. `fno_adv.csv (Symbol, 20Day_ADV)` - 4 edges
10. `Dynamic ADV Fallback via Yahoo Finance` - 4 edges

## Surprising Connections (you probably didn't know these)
- `trade_bot` --conceptually_related_to--> `Dual-Phase Institutional Pre-Open Suite`  [AMBIGUOUS]
  README.md → pre_open_monitor.py
- `Evening Phase - Harvest and Commit Rolling 20-Day ADV` --conceptually_related_to--> `NSE API Handshake (home then pre-open FO)`  [INFERRED]
  .github/workflows/main.yml → generate_adv.py
- `Evening Phase - Harvest and Commit Rolling 20-Day ADV` --conceptually_related_to--> `ADV Lookup from fno_adv.csv`  [INFERRED]
  .github/workflows/main.yml → pre_open_monitor.py
- `Morning Phase - Scan Pre-Open with Dynamic Offset` --conceptually_related_to--> `Shared fno_adv.csv Data Contract`  [INFERRED]
  .github/workflows/main.yml → pre_open_monitor.py
- `Morning Phase - Scan Pre-Open with Dynamic Offset` --references--> `Institutional Footprint Ratio Filter (>=5%)`  [INFERRED]
  .github/workflows/main.yml → pre_open_monitor.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Evening ADV Generation Pipeline** — _github_workflows_main_evening_phase, generate_adv_nse_handshake, generate_adv_yahoo_finance_adv, generate_adv_fno_adv_csv [INFERRED 0.95]
- **Morning Pre-Open Monitor and Email Alert Flow** — _github_workflows_main_morning_phase, pre_open_monitor_main, pre_open_monitor_nse_handshake, pre_open_monitor_adv_lookup, pre_open_monitor_footprint_ratio, pre_open_monitor_send_alert_email [INFERRED 0.95]
- **Shared fno_adv.csv Producer-Consumer Contract** — generate_adv_fno_adv_csv, pre_open_monitor_fno_adv_csv_contract, pre_open_monitor_adv_lookup, _github_workflows_main_evening_phase [INFERRED 0.95]

## Communities (5 total, 0 thin omitted)

### Community 0 - "GitHub Actions Orchestration"
Cohesion: 0.40
Nodes (6): ADV Calculation Cron (0 11 * * 1-5 UTC), Evening Phase - Harvest and Commit Rolling 20-Day ADV, execute_suite Job, GitHub Actions Schedule Branching, Dual-Phase Institutional Pre-Open Suite, trade_bot

### Community 1 - "Shared ADV Data Contract"
Cohesion: 0.60
Nodes (6): 45-Day Lookback for 20 Trading Sessions, fno_adv.csv (Symbol, 20Day_ADV), Yahoo Finance 20-Day ADV Calculation, ADV Lookup from fno_adv.csv, Dynamic ADV Fallback via Yahoo Finance, Shared fno_adv.csv Data Contract

### Community 2 - "NSE Pre-Open Signal Logic"
Cohesion: 0.53
Nodes (6): Active F&O Symbol Matrix, NSE API Handshake (home then pre-open FO), 5% Footprint Conviction Threshold, Institutional Footprint Ratio Filter (>=5%), pre_open_monitor __main__ Entrypoint, NSE Pre-Open FO API Handshake

### Community 3 - "Morning Monitor and Alerts"
Cohesion: 0.60
Nodes (4): Live Monitoring Cron (37 3 * * 1-5 UTC), Morning Phase - Scan Pre-Open with Dynamic Offset, send_alert_email(), SMTP Gmail Alert Path

## Ambiguous Edges - Review These
- `trade_bot` → `execute_suite Job`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to
- `trade_bot` → `Dual-Phase Institutional Pre-Open Suite`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to

## Knowledge Gaps
- **2 isolated node(s):** `ADV Calculation Cron (0 11 * * 1-5 UTC)`, `Live Monitoring Cron (37 3 * * 1-5 UTC)`
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `trade_bot` and `execute_suite Job`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `trade_bot` and `Dual-Phase Institutional Pre-Open Suite`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `Morning Phase - Scan Pre-Open with Dynamic Offset` connect `Morning Monitor and Alerts` to `Shared ADV Data Contract`, `NSE Pre-Open Signal Logic`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `Evening Phase - Harvest and Commit Rolling 20-Day ADV` connect `GitHub Actions Orchestration` to `Shared ADV Data Contract`, `NSE Pre-Open Signal Logic`, `Morning Monitor and Alerts`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `Morning Phase - Scan Pre-Open with Dynamic Offset` (e.g. with `Shared fno_adv.csv Data Contract` and `Institutional Footprint Ratio Filter (>=5%)`) actually correct?**
  _`Morning Phase - Scan Pre-Open with Dynamic Offset` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Evening Phase - Harvest and Commit Rolling 20-Day ADV` (e.g. with `Morning Phase - Scan Pre-Open with Dynamic Offset` and `NSE API Handshake (home then pre-open FO)`) actually correct?**
  _`Evening Phase - Harvest and Commit Rolling 20-Day ADV` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Yahoo Finance 20-Day ADV Calculation` (e.g. with `Dynamic ADV Fallback via Yahoo Finance` and `Institutional Footprint Ratio Filter (>=5%)`) actually correct?**
  _`Yahoo Finance 20-Day ADV Calculation` has 2 INFERRED edges - model-reasoned connections that need verification._
# trade_bot

NSE F&O pre-open institutional footprint monitor. GitHub Actions runs two weekday phases: evening ADV harvest, morning pre-open scan with email alerts.

## How it works

```mermaid
flowchart TB
  subgraph evening ["Evening · 16:30 IST"]
    A[NSE Pre-Open FO API] --> B[Active F&O symbols]
    B --> C[Yahoo Finance 45d bars]
    C --> D["20-day ADV mean"]
    D --> E[(fno_adv.csv)]
    E --> F[Commit CSV to repo]
  end

  subgraph morning ["Morning · 09:07 IST"]
    G[NSE Pre-Open FO API] --> H[IEP + matched volume]
    E --> I[ADV lookup]
    H --> J["Footprint% = Matched_Vol / ADV × 100"]
    I --> J
    J --> K{Footprint ≥ 5%?}
    K -->|yes| L[SMTP email alert]
    K -->|no| M[Suppress email]
  end

  F -.->|next morning| I
```

| Piece | Role |
|---|---|
| `generate_adv.py` | Builds `fno_adv.csv` (Symbol → 20Day_ADV) |
| `pre_open_monitor.py` | Live pre-open scan + ≥5% footprint filter + email |
| `.github/workflows/main.yml` | Cron orchestration for both phases |
| `graphify-out/` | Knowledge graph of architecture (`graph.html`, `GRAPH_REPORT.md`) |

## Secrets

Set GitHub Actions secrets: `SENDER_EMAIL`, `SENDER_PASSWORD`, `RECEIVER_EMAIL`.

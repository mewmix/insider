# Polymarket Whale Scanner & Analytics

This directory contains tools for monitoring Polymarket activity, tracking whales, and analyzing PnL.

## Tools

- `app.py`: FastAPI service that scans Polymarket for large trades ("whales") and sends alerts to Telegram. Also provides an API for reports.
- `subgraph_scanner.py`: Scans a specific wallet's history from the subgraph to build a local DB of their positions, PnL, and order fills.
- `pnl_leaderboard.py`: Fetches top PnL accounts from the subgraph and verifies their net PnL.
- `active_positions_report.py`: Analyzes active positions of top traders to find overlaps or contrary bets.

## Data

- `scanner.db`: SQLite database for the whale scanner (stores seen trades, wallets, funding info).
- `subgraph_scanner.db`: SQLite database for the individual wallet scanner.

## Usage

### Whale Scanner (Service)

1.  Set environment variables (see `app.py` or `.env.example`). Key ones:
    -   `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
    -   `ALCHEMY_RPC_URL` (optional, for funding source check)
2.  Run:
    ```bash
    python3 app.py
    ```

### Wallet PnL Scanner

```bash
python3 subgraph_scanner.py --address <WALLET_ADDRESS> --follow
```

### Leaderboard

```bash
python3 pnl_leaderboard.py
```

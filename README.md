# Polymarket Fresh Whale Scanner

Minimal FastAPI service that polls Polymarket Data API for large trades, checks whether the wallet is fresh using Data API activity, and posts Telegram alerts.

## What it exposes

- `GET /healthz`
- `POST /scan/once`
- `POST /scan/start`
- `POST /scan/stop`
- `GET /report/insiders` (fresh wallets, smart insiders, suspicious accounts)

`/report/insiders` includes a `meta` block describing the thresholds used for each category.

## Runbook

### 1) Create a Telegram bot + chat id

1. Create a bot with BotFather and get `TELEGRAM_BOT_TOKEN`.
2. Get your chat id:
   - Send the bot a message and use `getUpdates`, or
   - Add the bot to a private channel and use that channel id.
3. Set `TELEGRAM_CHAT_ID`.

### 2) Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Create `.env` from `.env.example` and fill in values.

### 3) Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

### 4) Validate

```bash
curl -s http://localhost:8080/healthz
curl -s -X POST http://localhost:8080/scan/once
curl -s -X POST http://localhost:8080/scan/start
```

### 5) Deploy (cheap)

- Fly.io / Railway / Render for a tiny always-on container.
- A small VPS for fixed cost.

If you want cheaper, run `/scan/once` on a schedule (cron) and skip the long-running process.

## Design notes

- Trade scanning uses Data API `/trades` with `filterType=CASH&filterAmount=<BIG_USDC>`.
- Freshness uses Data API `/activity` with `sortDirection=ASC` to estimate first-seen timestamp.
- Optional enrichment from Gamma API if you want richer market metadata in alerts.

## Configuration

See `.env.example` for defaults.

Key link configuration:

- `POLYMARKET_MARKET_BASE` for market links (default `https://polymarket.com/market`)
- `EXPLORER_TX_BASE` for tx links (default `https://polygonscan.com/tx`)
- `EXPLORER_ADDRESS_BASE` for address links (default `https://polygonscan.com/address`)
- `MARKET_REFRESH_SECONDS` for Gamma metadata refresh interval
- `MARKET_POLL_SECONDS` for the background resolution poll interval
- `ALCHEMY_RPC_URL` for funding timestamp lookups
- `FUNDED_MAX_AGE_SECONDS` and `FUNDED_MIN_NOTIONAL` for suspicious accounts
- `FUNDING_POLL_SECONDS` and `FUNDING_REFRESH_SECONDS` for funding refresh cadence

## Notes

- This uses Data API polling; it does not rely on the authenticated CLOB websocket user channel.
- Use `BIG_USDC`, `FRESH_MAX_AGE_SECONDS`, and `POLL_SECONDS` to tune sensitivity and load.
- Gamma metadata is cached in the `markets` table and used for resolution tracking and alert enrichment.
- Alerts include funding age when `ALCHEMY_RPC_URL` is set.

## Subgraph scanner (deeper data)

The script `subgraph_scanner.py` scans the public Polymarket subgraphs to pull
order fills, map token IDs to condition IDs, and summarize PNL/redemptions/open interest.

Example:

```bash
python3 subgraph_scanner.py --address 0xd90edE33043f26859ADb1Dcbd79d45EB125d1aB3
```

You can also follow continuously:

```bash
python3 subgraph_scanner.py --address 0xd90edE33043f26859ADb1Dcbd79d45EB125d1aB3 --follow
```

## Active positions report

`active_positions_report.py` builds a daily snapshot of active traders and their
current positions, plus overlap/contrary summaries by market.

Example:

```bash
python3 active_positions_report.py --with-metadata
```

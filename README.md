# Polymarket Fresh Whale Scanner

Minimal FastAPI service that polls Polymarket Data API for large trades, checks whether the wallet is fresh using Data API activity, and posts Telegram alerts.

## What it exposes

- `GET /healthz`
- `POST /scan/once`
- `POST /scan/start`
- `POST /scan/stop`

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

## Notes

- This uses Data API polling; it does not rely on the authenticated CLOB websocket user channel.
- Use `BIG_USDC`, `FRESH_MAX_AGE_SECONDS`, and `POLL_SECONDS` to tune sensitivity and load.

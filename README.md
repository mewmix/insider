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

## Arbitrum v2 skim scanner

`skim_scanner.py` scans Camelot v2 and Uniswap v2 pairs on Arbitrum for skim
opportunities by comparing reserves to actual token balances.

Example:

```bash
python3 skim_scanner.py --dex camelot --max-pairs 200 --min-imbalance 0.05
```

To crawl and persist all pairs into a sqlite db (batch size 1000, resume supported):

```bash
python3 skim_scanner.py --dex both --crawl --batch-size 1000 --max-pairs 0 --resume
```

To scan pairs directly from the sqlite db:

```bash
python3 skim_scanner.py --dex both --scan-db --max-pairs 0 --rotate-rpc
```

## Flash swap scanner

`flash_swap_scanner.py` loads Camelot + Uniswap v2 pairs from `skim_pairs.db`,
pulls reserves on-chain, and simulates optimal two-hop arbitrage to estimate
potential profit (no transactions sent).
Default fees: Camelot/Sushi 0.5% (`--fee-camelot`, `--fee-sushiswap`), Uniswap v2 0.3%.

Example:

```bash
python3 flash_swap_scanner.py --top 25 --min-profit 0.01 --max-trade-frac 0.3
```

The scanner also estimates USD profit using on-chain reserves (stablecoins/WETH).
You can focus on high-liquidity and high-volume pairs:

```bash
python3 flash_swap_scanner.py --focus-top-reserve 200 --focus-top-volume 200
```

Scan multiple v2 DEXes (default: uniswapv2, camelot, sushiswapv2):

```bash
python3 flash_swap_scanner.py --dexes uniswapv2,camelot,sushiswapv2
```

## Flash swap gas simulation

`flash_swap_sim.py` compiles a minimal flash-swap executor contract and can
estimate deployment + execution gas. Use `--deploy` to deploy and then estimate
`execute` gas.

Example (gas estimate only, no deploy):

```bash
python3 flash_swap_sim.py --pair-borrow <PAIR> --pair-swap <PAIR> --token-borrow <TOKEN> --amount-borrow <RAW>
```

Example (deploy + estimate execute):

```bash
python3 flash_swap_sim.py --deploy --pair-borrow <PAIR> --pair-swap <PAIR> --token-borrow <TOKEN> --amount-borrow <RAW>
```

Environment overrides:

- `ARBITRUM_RPC_URL` (defaults to a public Arbitrum RPC)
- `ARBITRUM_RPC_URLS` (comma-separated RPC URLs to rotate across)
- `CAMELOT_V2_SUBGRAPH`
- `UNISWAP_V2_SUBGRAPH`
- `SUSHISWAP_V2_SUBGRAPH`
- `GRAPH_API_KEY` (required for The Graph gateway endpoints)

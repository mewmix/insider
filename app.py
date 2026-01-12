import asyncio
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import html
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

# ---- Config ----

DATA_API_BASE = os.getenv("DATA_API_BASE", "https://data-api.polymarket.com")
GAMMA_API_BASE = os.getenv("GAMMA_API_BASE", "https://gamma-api.polymarket.com")
POLYMARKET_MARKET_BASE = os.getenv("POLYMARKET_MARKET_BASE", "https://polymarket.com/market")
EXPLORER_TX_BASE = os.getenv("EXPLORER_TX_BASE", "https://polygonscan.com/tx")
EXPLORER_ADDRESS_BASE = os.getenv("EXPLORER_ADDRESS_BASE", "https://polygonscan.com/address")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "15"))
BIG_USDC = float(os.getenv("BIG_USDC", "5000"))
FRESH_MAX_AGE_SECONDS = int(os.getenv("FRESH_MAX_AGE_SECONDS", "259200"))
FRESH_MAX_FIRST_SEEN_ONLY = os.getenv("FRESH_MAX_FIRST_SEEN_ONLY", "true").lower() == "true"
MARKET_REFRESH_SECONDS = int(os.getenv("MARKET_REFRESH_SECONDS", "3600"))
MARKET_POLL_SECONDS = int(os.getenv("MARKET_POLL_SECONDS", "300"))
POLYGONSCAN_API_KEY = os.getenv("POLYGONSCAN_API_KEY", "").strip()
FUNDED_MAX_AGE_SECONDS = int(os.getenv("FUNDED_MAX_AGE_SECONDS", "604800"))
FUNDED_MIN_NOTIONAL = float(os.getenv("FUNDED_MIN_NOTIONAL", "10000"))
FUNDING_POLL_SECONDS = int(os.getenv("FUNDING_POLL_SECONDS", "900"))
FUNDING_REFRESH_SECONDS = int(os.getenv("FUNDING_REFRESH_SECONDS", "86400"))

SQLITE_PATH = os.getenv("SQLITE_PATH", "scanner.db")

# ---- Persistence (SQLite) ----


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    conn = db()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_trades (
              tx_hash TEXT PRIMARY KEY,
              ts INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
              tx_hash TEXT PRIMARY KEY,
              proxy_wallet TEXT NOT NULL,
              side TEXT,
              condition_id TEXT,
              size REAL,
              price REAL,
              timestamp INTEGER,
              title TEXT,
              slug TEXT,
              outcome TEXT,
              outcome_index INTEGER,
              notional_usdc REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS wallets (
              address TEXT PRIMARY KEY,
              first_seen_ts INTEGER,
              label TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS wallet_funding (
              address TEXT PRIMARY KEY,
              first_funded_ts INTEGER,
              source TEXT,
              updated_at_ts INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kv (
              k TEXT PRIMARY KEY,
              v TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS markets (
              condition_id TEXT PRIMARY KEY,
              slug TEXT,
              title TEXT,
              category TEXT,
              icon TEXT,
              outcomes_json TEXT,
              outcome_prices_json TEXT,
              winner_outcome_index INTEGER,
              winner_outcome TEXT,
              resolution_status TEXT,
              resolved_by TEXT,
              resolved INTEGER NOT NULL DEFAULT 0,
              updated_at_ts INTEGER
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def seen_tx(tx_hash: str) -> bool:
    conn = db()
    try:
        # Check both old seen_trades and new trades table
        row = conn.execute("SELECT 1 FROM seen_trades WHERE tx_hash = ?", (tx_hash,)).fetchone()
        if row:
            return True
        row = conn.execute("SELECT 1 FROM trades WHERE tx_hash = ?", (tx_hash,)).fetchone()
        return row is not None
    finally:
        conn.close()


def save_trade(t: Any) -> None:
    # t is a Trade dataclass
    conn = db()
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO trades (
                tx_hash, proxy_wallet, side, condition_id, size, price, 
                timestamp, title, slug, outcome, outcome_index, notional_usdc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                t.tx_hash,
                t.proxy_wallet,
                t.side,
                t.condition_id,
                t.size,
                t.price,
                t.timestamp,
                t.title,
                t.slug,
                t.outcome,
                t.outcome_index,
                t.notional_usdc_est,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def mark_tx(tx_hash: str, ts: int) -> None:
    conn = db()
    try:
        conn.execute("INSERT OR IGNORE INTO seen_trades(tx_hash, ts) VALUES(?, ?)", (tx_hash, ts))
        conn.commit()
    finally:
        conn.close()


def kv_get(key: str) -> Optional[str]:
    conn = db()
    try:
        row = conn.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def kv_set(key: str, val: str) -> None:
    conn = db()
    try:
        conn.execute(
            "INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (key, val),
        )
        conn.commit()
    finally:
        conn.close()


# ---- Telegram ----


async def telegram_send(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
        "parse_mode": "HTML",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()


# ---- Polymarket clients ----


@dataclass(frozen=True)
class Trade:
    proxy_wallet: str
    side: str
    condition_id: str
    size: float
    price: float
    timestamp: int
    title: str
    slug: str
    outcome: str
    outcome_index: int
    tx_hash: str

    @property
    def notional_usdc_est(self) -> float:
        return float(self.size) * float(self.price)


async def fetch_trades(limit: int = 100) -> List[Trade]:
    params = {
        "limit": str(limit),
        "offset": "0",
        "takerOnly": "true",
        "filterType": "CASH",
        "filterAmount": str(BIG_USDC),
    }

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(f"{DATA_API_BASE}/trades", params=params)
        r.raise_for_status()
        data = r.json()

    out: List[Trade] = []
    for row in data:
        txh = str(row.get("transactionHash", "")).strip()
        if not txh:
            continue

        out.append(
            Trade(
                proxy_wallet=str(row.get("proxyWallet", "")).lower(),
                side=str(row.get("side", "")),
                condition_id=str(row.get("conditionId", "")),
                size=float(row.get("size", 0.0) or 0.0),
                price=float(row.get("price", 0.0) or 0.0),
                timestamp=int(row.get("timestamp", 0) or 0),
                title=str(row.get("title", "")),
                slug=str(row.get("slug", "")),
                outcome=str(row.get("outcome", "")),
                outcome_index=int(row.get("outcomeIndex", -1) or -1),
                tx_hash=txh,
            )
        )
    return out


async def fetch_first_seen_ts(proxy_wallet: str) -> Optional[int]:
    params = {
        "user": proxy_wallet,
        "type": "TRADE",
        "limit": "1",
        "offset": "0",
        "sortBy": "TIMESTAMP",
        "sortDirection": "ASC",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(f"{DATA_API_BASE}/activity", params=params)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()

    if not data:
        return None
    return int(data[0].get("timestamp", 0) or 0)


def is_fresh(first_seen_ts: Optional[int], now_ts: int) -> bool:
    if first_seen_ts is None:
        return True
    age = now_ts - first_seen_ts
    return age <= FRESH_MAX_AGE_SECONDS


def parse_json_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [str(x) for x in data]
    return []


def infer_winner_index(outcomes: List[str], outcome_prices: List[str]) -> Optional[int]:
    if not outcomes or not outcome_prices or len(outcomes) != len(outcome_prices):
        return None
    try:
        prices = [float(x) for x in outcome_prices]
    except ValueError:
        return None
    max_price = max(prices)
    if max_price < 0.99:
        return None
    return prices.index(max_price)


def get_market_metadata(condition_id: str) -> Optional[Dict[str, Any]]:
    conn = db()
    try:
        row = conn.execute(
            """
            SELECT slug, title, category, icon, outcomes_json, outcome_prices_json,
                   winner_outcome_index, winner_outcome, resolution_status, resolved,
                   updated_at_ts
            FROM markets
            WHERE condition_id = ?
            """,
            (condition_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "slug": row[0],
            "title": row[1],
            "category": row[2],
            "icon": row[3],
            "outcomes_json": row[4],
            "outcome_prices_json": row[5],
            "winner_outcome_index": row[6],
            "winner_outcome": row[7],
            "resolution_status": row[8],
            "resolved": bool(row[9]),
            "updated_at_ts": row[10],
        }
    finally:
        conn.close()


async def fetch_market_by_condition_id(condition_id: str) -> Optional[Dict[str, Any]]:
    params = {"conditionId": condition_id}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(f"{GAMMA_API_BASE}/markets", params=params)
        r.raise_for_status()
        data = r.json()
    if not data:
        return None
    return data[0]


async def upsert_market_from_gamma(condition_id: str) -> Optional[Dict[str, Any]]:
    market = await fetch_market_by_condition_id(condition_id)
    if not market:
        return None

    outcomes_raw = str(market.get("outcomes") or "")
    outcome_prices_raw = str(market.get("outcomePrices") or "")
    outcomes = parse_json_list(outcomes_raw)
    outcome_prices = parse_json_list(outcome_prices_raw)
    winner_index = infer_winner_index(outcomes, outcome_prices)
    winner_outcome = outcomes[winner_index] if winner_index is not None else None
    resolved = 1 if winner_index is not None else 0

    conn = db()
    try:
        conn.execute(
            """
            INSERT INTO markets (
              condition_id, slug, title, category, icon, outcomes_json, outcome_prices_json,
              winner_outcome_index, winner_outcome, resolution_status, resolved_by, resolved,
              updated_at_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(condition_id) DO UPDATE SET
              slug=excluded.slug,
              title=excluded.title,
              category=excluded.category,
              icon=excluded.icon,
              outcomes_json=excluded.outcomes_json,
              outcome_prices_json=excluded.outcome_prices_json,
              winner_outcome_index=excluded.winner_outcome_index,
              winner_outcome=excluded.winner_outcome,
              resolution_status=excluded.resolution_status,
              resolved_by=excluded.resolved_by,
              resolved=excluded.resolved,
              updated_at_ts=excluded.updated_at_ts
            """,
            (
                condition_id,
                market.get("slug"),
                market.get("question") or market.get("title"),
                market.get("category"),
                market.get("icon"),
                outcomes_raw,
                outcome_prices_raw,
                winner_index,
                winner_outcome,
                market.get("umaResolutionStatus"),
                market.get("resolvedBy"),
                resolved,
                int(time.time()),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return get_market_metadata(condition_id)


async def ensure_market_metadata(condition_id: str) -> Optional[Dict[str, Any]]:
    row = get_market_metadata(condition_id)
    if row is None:
        return await upsert_market_from_gamma(condition_id)
    updated_at_ts = row.get("updated_at_ts")
    if updated_at_ts is None:
        return await upsert_market_from_gamma(condition_id)
    if (int(time.time()) - int(updated_at_ts)) >= MARKET_REFRESH_SECONDS:
        return await upsert_market_from_gamma(condition_id)
    return row


async def fetch_first_funded_ts(address: str) -> Optional[int]:
    if not POLYGONSCAN_API_KEY:
        return None
    params = {
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": "0",
        "endblock": "99999999",
        "page": "1",
        "offset": "100",
        "sort": "asc",
        "apikey": POLYGONSCAN_API_KEY,
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get("https://api.polygonscan.com/api", params=params)
        r.raise_for_status()
        data = r.json()
    if data.get("status") == "0" and "No transactions" in str(data.get("message", "")):
        return None
    result = data.get("result") or []
    if not isinstance(result, list) or not result:
        return None
    address_l = address.lower()
    first_inbound = None
    for tx in result:
        to_addr = str(tx.get("to", "")).lower()
        value = int(tx.get("value", "0") or 0)
        if to_addr == address_l and value > 0:
            first_inbound = tx
            break
    tx = first_inbound or result[0]
    ts = int(tx.get("timeStamp", 0) or 0)
    return ts if ts > 0 else None


def upsert_wallet_funding(address: str, first_funded_ts: Optional[int]) -> None:
    conn = db()
    try:
        conn.execute(
            """
            INSERT INTO wallet_funding (
              address, first_funded_ts, source, updated_at_ts
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(address) DO UPDATE SET
              first_funded_ts=excluded.first_funded_ts,
              source=excluded.source,
              updated_at_ts=excluded.updated_at_ts
            """,
            (address, first_funded_ts, "polygonscan", int(time.time())),
        )
        conn.commit()
    finally:
        conn.close()


def format_alert(
    t: Trade,
    first_seen_ts: Optional[int],
    market_meta: Optional[Dict[str, Any]],
) -> str:
    now_ts = int(time.time())
    age_s = None if first_seen_ts is None else max(0, now_ts - first_seen_ts)
    age_h = None if age_s is None else round(age_s / 3600, 2)

    market_url = f"{POLYMARKET_MARKET_BASE}/{t.slug}" if t.slug else POLYMARKET_MARKET_BASE
    tx_url = f"{EXPLORER_TX_BASE}/{t.tx_hash}"
    addr_url = f"{EXPLORER_ADDRESS_BASE}/{t.proxy_wallet}"

    def link(label: str, url: str) -> str:
        return f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a>'

    lines = [
        "🚨 <b>" + link("Polymarket Whale Alert", tx_url) + "</b> 🚨",
        "👤 " + link(f"Wallet: {t.proxy_wallet}", addr_url),
        "🔮 " + link(f"Market: {t.title}", market_url),
        "⚖️ " + link(f"Side: {t.side} | Outcome: {t.outcome}", tx_url),
        "💰 " + link(f"Value: ${t.notional_usdc_est:,.2f}", tx_url),
        "🧾 " + link(f"Tx: {t.tx_hash}", tx_url),
    ]
    if market_meta:
        if market_meta.get("category"):
            lines.append("🏷️ " + html.escape(f"Category: {market_meta['category']}"))
        if market_meta.get("icon"):
            lines.append("🖼️ " + link("Icon", market_meta["icon"]))
    if age_h is None:
        age_label = "First-seen: unknown"
    else:
        age_label = f"First-seen: {age_h}h ago"
    lines.append("⏳ " + link(age_label, tx_url))
    return "\n".join(lines)


# ---- Scanner loop ----


scanner_task: Optional[asyncio.Task] = None
scanner_stop_evt = asyncio.Event()
market_sync_task: Optional[asyncio.Task] = None
market_sync_stop_evt = asyncio.Event()
funding_sync_task: Optional[asyncio.Task] = None
funding_sync_stop_evt = asyncio.Event()


def get_conditions_needing_refresh(limit: int) -> List[str]:
    now_ts = int(time.time())
    cutoff_ts = now_ts - MARKET_REFRESH_SECONDS
    conn = db()
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT t.condition_id
            FROM trades t
            LEFT JOIN markets m ON t.condition_id = m.condition_id
            WHERE m.condition_id IS NULL
               OR m.resolved = 0
               OR m.updated_at_ts IS NULL
               OR m.updated_at_ts < ?
            LIMIT ?
            """,
            (cutoff_ts, limit),
        ).fetchall()
        return [r[0] for r in rows if r[0]]
    finally:
        conn.close()


async def market_sync_loop() -> None:
    market_sync_stop_evt.clear()
    while not market_sync_stop_evt.is_set():
        condition_ids = get_conditions_needing_refresh(limit=50)
        for condition_id in condition_ids:
            try:
                await upsert_market_from_gamma(condition_id)
            except Exception:
                continue
        await asyncio.sleep(MARKET_POLL_SECONDS)


def get_wallets_needing_funding(limit: int) -> List[str]:
    now_ts = int(time.time())
    cutoff_ts = now_ts - FUNDING_REFRESH_SECONDS
    conn = db()
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT t.proxy_wallet
            FROM trades t
            LEFT JOIN wallet_funding f ON t.proxy_wallet = f.address
            WHERE f.address IS NULL
               OR f.updated_at_ts IS NULL
               OR f.updated_at_ts < ?
            LIMIT ?
            """,
            (cutoff_ts, limit),
        ).fetchall()
        return [r[0] for r in rows if r[0]]
    finally:
        conn.close()


async def funding_sync_loop() -> None:
    funding_sync_stop_evt.clear()
    while not funding_sync_stop_evt.is_set():
        if not POLYGONSCAN_API_KEY:
            await asyncio.sleep(FUNDING_POLL_SECONDS)
            continue
        addresses = get_wallets_needing_funding(limit=25)
        for address in addresses:
            try:
                funded_ts = await fetch_first_funded_ts(address)
                upsert_wallet_funding(address, funded_ts)
            except Exception:
                continue
        await asyncio.sleep(FUNDING_POLL_SECONDS)


async def scan_once() -> Dict[str, Any]:
    now_ts = int(time.time())

    trades = await fetch_trades(limit=100)

    last_ts_str = kv_get("last_seen_trade_ts")
    last_seen_ts = int(last_ts_str) if last_ts_str else 0

    new_max_ts = last_seen_ts
    emitted = 0
    considered = 0

    trades_sorted = sorted(trades, key=lambda x: (x.timestamp, x.tx_hash))

    for t in trades_sorted:
        if t.timestamp <= last_seen_ts:
            continue

        new_max_ts = max(new_max_ts, t.timestamp)
        if seen_tx(t.tx_hash):
            continue
        
        # Save every trade we see for historical analysis
        save_trade(t)

        considered += 1

        if t.notional_usdc_est < BIG_USDC:
            # We already saved it to 'trades', so just ensuring we don't re-process
            mark_tx(t.tx_hash, t.timestamp)
            continue

        first_seen = await fetch_first_seen_ts(t.proxy_wallet)
        
        # Update wallet DB
        if first_seen is not None:
            conn = db()
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO wallets(address, first_seen_ts) VALUES(?, ?)",
                    (t.proxy_wallet, first_seen),
                )
                conn.commit()
            finally:
                conn.close()

        fresh = is_fresh(first_seen, now_ts)

        if FRESH_MAX_FIRST_SEEN_ONLY and not fresh:
            mark_tx(t.tx_hash, t.timestamp)
            continue

        try:
            market_meta = await ensure_market_metadata(t.condition_id)
        except Exception:
            market_meta = None
        await telegram_send(format_alert(t, first_seen, market_meta))
        emitted += 1
        mark_tx(t.tx_hash, t.timestamp)

    if new_max_ts > last_seen_ts:
        kv_set("last_seen_trade_ts", str(new_max_ts))

    return {
        "considered": considered,
        "emitted": emitted,
        "last_seen_ts": last_seen_ts,
        "new_last_seen_ts": new_max_ts,
        "polled": len(trades),
    }


async def scanner_loop() -> None:
    scanner_stop_evt.clear()
    while not scanner_stop_evt.is_set():
        try:
            await scan_once()
        except Exception as e:
            try:
                await telegram_send(f"Polymarket scanner error: {type(e).__name__}: {e}")
            except Exception:
                pass
        await asyncio.sleep(POLL_SECONDS)


# ---- FastAPI app ----


init_db()
app = FastAPI(title="Polymarket Fresh Whale Scanner")


class ScanResponse(BaseModel):
    considered: int
    emitted: int
    last_seen_ts: int
    new_last_seen_ts: int
    polled: int


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return {
        "ok": True,
        "poll_seconds": POLL_SECONDS,
        "big_usdc": BIG_USDC,
        "fresh_max_age_seconds": FRESH_MAX_AGE_SECONDS,
        "data_api_base": DATA_API_BASE,
        "gamma_api_base": GAMMA_API_BASE,
        "scanner_running": scanner_task is not None and not scanner_task.done(),
        "market_sync_running": market_sync_task is not None and not market_sync_task.done(),
        "funding_sync_running": funding_sync_task is not None and not funding_sync_task.done(),
    }


@app.on_event("startup")
async def startup_tasks() -> None:
    global market_sync_task
    global funding_sync_task
    if market_sync_task is None or market_sync_task.done():
        market_sync_task = asyncio.create_task(market_sync_loop())
    if funding_sync_task is None or funding_sync_task.done():
        funding_sync_task = asyncio.create_task(funding_sync_loop())


@app.post("/scan/once", response_model=ScanResponse)
async def scan_once_endpoint() -> ScanResponse:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise HTTPException(status_code=500, detail="Missing Telegram env vars")
    res = await scan_once()
    return ScanResponse(**res)


@app.post("/scan/start")
async def scan_start() -> Dict[str, Any]:
    global scanner_task
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise HTTPException(status_code=500, detail="Missing Telegram env vars")

    if scanner_task is not None and not scanner_task.done():
        return {"started": False, "reason": "already running"}

    scanner_task = asyncio.create_task(scanner_loop())
    return {"started": True}


@app.post("/scan/stop")
async def scan_stop() -> Dict[str, Any]:
    global scanner_task
    scanner_stop_evt.set()
    if scanner_task:
        await asyncio.sleep(0)
    return {"stopped": True}


@app.post("/scan/test")
async def scan_test() -> Dict[str, Any]:
    await telegram_send(
        format_alert(
            Trade(
                proxy_wallet="0xd1d51797c9d038af763bbc313c2aacdc4ed08845",
                side="BUY",
                condition_id="0xd47d23abc3dcdcfcf7eca3b88d7ff5769fd143e90c7cce4de2a2380869900b2e",
                size=1000.0,
                price=81.01575,
                timestamp=int(time.time()),
                title="Bitcoin Up or Down - January 11, 6:00PM-6:15PM ET",
                slug="btc-updown-15m-1768172400",
                outcome="Up",
                outcome_index=-1,
                tx_hash="0x9a395b45127220c9a8815caf0aa8fb458362a7d34f0b5f37d63390f6b20b71b2",
            ),
            int(time.time()) - 38 * 3600,
            None,
        )
    )
    return {"sent": True}


@app.get("/report/insiders")
def report_insiders() -> Dict[str, Any]:
    """
    Experimental: Find wallets that have 'high win rate' or 'high profit' if we could calculate it.
    For now, let's find wallets with high volume or frequent large trades that are 'fresh'.
    """
    conn = db()
    try:
        now_ts = int(time.time())
        fresh_rows = conn.execute(
            """
            SELECT 
                w.address,
                w.first_seen_ts,
                COUNT(t.tx_hash) as trade_count,
                SUM(t.notional_usdc) as total_vol
            FROM wallets w
            LEFT JOIN trades t ON t.proxy_wallet = w.address
            WHERE w.first_seen_ts IS NOT NULL
              AND w.first_seen_ts >= ?
            GROUP BY w.address
            HAVING trade_count > 0
            ORDER BY w.first_seen_ts DESC, total_vol DESC
            LIMIT 50
            """,
            (now_ts - FRESH_MAX_AGE_SECONDS,),
        ).fetchall()

        smart_rows = conn.execute(
            """
            SELECT 
                t.proxy_wallet,
                COUNT(*) as trade_count,
                SUM(t.notional_usdc) as total_vol,
                w.first_seen_ts,
                SUM(CASE WHEN m.winner_outcome_index IS NOT NULL THEN 1 ELSE 0 END) as resolved_trades,
                SUM(
                  CASE
                    WHEN m.winner_outcome_index IS NOT NULL
                     AND t.outcome_index = m.winner_outcome_index
                    THEN 1 ELSE 0 END
                ) as wins,
                SUM(
                  CASE
                    WHEN m.winner_outcome_index IS NULL THEN 0
                    WHEN t.side = 'BUY' AND t.outcome_index = m.winner_outcome_index
                      THEN t.size * (1 - t.price)
                    WHEN t.side = 'BUY'
                      THEN -t.size * t.price
                    WHEN t.side = 'SELL' AND t.outcome_index = m.winner_outcome_index
                      THEN -t.size * (1 - t.price)
                    WHEN t.side = 'SELL'
                      THEN t.size * t.price
                    ELSE 0
                  END
                ) as realized_pnl
            FROM trades t
            LEFT JOIN wallets w ON t.proxy_wallet = w.address
            LEFT JOIN markets m ON t.condition_id = m.condition_id
            GROUP BY t.proxy_wallet
            HAVING trade_count > 0
            """
        ).fetchall()

        suspicious_rows = conn.execute(
            """
            SELECT
                t.proxy_wallet,
                w.first_seen_ts,
                f.first_funded_ts,
                COUNT(*) as trade_count,
                SUM(t.notional_usdc) as total_vol,
                MAX(t.notional_usdc) as max_trade
            FROM trades t
            LEFT JOIN wallets w ON t.proxy_wallet = w.address
            LEFT JOIN wallet_funding f ON t.proxy_wallet = f.address
            WHERE f.first_funded_ts IS NOT NULL
              AND f.first_funded_ts >= ?
            GROUP BY t.proxy_wallet
            HAVING max_trade >= ?
            ORDER BY max_trade DESC, total_vol DESC
            LIMIT 50
            """,
            (now_ts - FUNDED_MAX_AGE_SECONDS, FUNDED_MIN_NOTIONAL),
        ).fetchall()

        fresh_wallets = []
        for r in fresh_rows:
            fresh_wallets.append(
                {
                    "wallet": r[0],
                    "first_seen_ts": r[1],
                    "count": r[2],
                    "volume": r[3],
                }
            )

        smart_insiders = []
        for r in smart_rows:
            resolved_trades = int(r[4] or 0)
            wins = int(r[5] or 0)
            win_rate = (wins / resolved_trades) if resolved_trades > 0 else None
            realized_pnl = float(r[6] or 0.0)
            if resolved_trades < 2 and realized_pnl < 1000.0:
                continue
            smart_insiders.append(
                {
                    "wallet": r[0],
                    "count": r[1],
                    "volume": r[2],
                    "first_seen_ts": r[3],
                    "resolved_trades": resolved_trades,
                    "wins": wins,
                    "win_rate": win_rate,
                    "realized_pnl_est": realized_pnl,
                }
            )
        smart_insiders.sort(
            key=lambda x: (
                x["win_rate"] if x["win_rate"] is not None else -1.0,
                x["realized_pnl_est"],
            ),
            reverse=True,
        )
        smart_insiders = smart_insiders[:50]

        suspicious_accounts = []
        for r in suspicious_rows:
            suspicious_accounts.append(
                {
                    "wallet": r[0],
                    "first_seen_ts": r[1],
                    "first_funded_ts": r[2],
                    "count": r[3],
                    "volume": r[4],
                    "max_trade": r[5],
                }
            )

        return {
            "meta": {
                "fresh_wallets": {
                    "first_seen_within_seconds": FRESH_MAX_AGE_SECONDS,
                    "description": "Wallets first seen via Polymarket activity within the freshness window.",
                },
                "smart_insiders": {
                    "min_resolved_trades": 2,
                    "min_realized_pnl": 1000,
                    "rank": "win_rate desc, realized_pnl_est desc",
                    "description": "Wallets with resolved outcomes or high realized PnL.",
                },
                "suspicious_accounts": {
                    "funded_within_seconds": FUNDED_MAX_AGE_SECONDS,
                    "min_single_trade_notional": FUNDED_MIN_NOTIONAL,
                    "description": "Recently funded wallets placing large bets.",
                },
            },
            "fresh_wallets": fresh_wallets,
            "smart_insiders": smart_insiders,
            "suspicious_accounts": suspicious_accounts,
        }
    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

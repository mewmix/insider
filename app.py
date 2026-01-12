import asyncio
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
            CREATE TABLE IF NOT EXISTS kv (
              k TEXT PRIMARY KEY,
              v TEXT NOT NULL
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


def format_alert(t: Trade, first_seen_ts: Optional[int]) -> str:
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
    if age_h is None:
        age_label = "First-seen: unknown"
    else:
        age_label = f"First-seen: {age_h}h ago"
    lines.append("⏳ " + link(age_label, tx_url))
    return "\n".join(lines)


# ---- Scanner loop ----


scanner_task: Optional[asyncio.Task] = None
scanner_stop_evt = asyncio.Event()


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

        await telegram_send(format_alert(t, first_seen))
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
    }


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
        # Example query: Wallets with > 1 trade, sorted by total volume
        rows = conn.execute(
            """
            SELECT 
                t.proxy_wallet, 
                COUNT(*) as trade_count, 
                SUM(t.notional_usdc) as total_vol,
                w.first_seen_ts
            FROM trades t
            LEFT JOIN wallets w ON t.proxy_wallet = w.address
            GROUP BY t.proxy_wallet
            HAVING trade_count > 0
            ORDER BY total_vol DESC
            LIMIT 50
            """
        ).fetchall()
        
        results = []
        for r in rows:
            results.append({
                "wallet": r[0],
                "count": r[1],
                "volume": r[2],
                "first_seen_ts": r[3]
            })
        return {"insiders": results}
    finally:
        conn.close()

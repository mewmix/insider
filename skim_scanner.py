import argparse
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import Dict, Iterable, List, Optional, Tuple

import httpx
from dotenv import load_dotenv


load_dotenv()

getcontext().prec = 50

ARBITRUM_RPC_URL = os.getenv("ARBITRUM_RPC_URL", "https://arb1.arbitrum.io/rpc")

RPC_ENDPOINTS = {
    "drpc_ws": "wss://arbitrum.drpc.org",
    "1rpc": "https://1rpc.io/arb",
    "publicnode_ws": "wss://arbitrum-one-rpc.publicnode.com",
    "zan": "https://api.zan.top/arb-one",
    "drpc": "https://arbitrum.drpc.org",
    "fastnode": "https://public-arb-mainnet.fastnode.io",
    "owlracle": "https://rpc.owlracle.info/arb/70d38ce1826c4a60bb2a8e05a6c8b20f",
    "nodies": "https://arbitrum-one-public.nodies.app",
    "publicnode": "https://arbitrum-one-rpc.publicnode.com",
    "tatum": "https://arb-one-mainnet.gateway.tatum.io",
    "tenderly": "https://arbitrum.gateway.tenderly.co",
    "lava": "https://arb1.lava.build",
    "blast": "https://arbitrum-one.public.blastapi.io",
    "subquery": "https://arbitrum.rpc.subquery.network/public",
    "blockpi": "https://arbitrum.public.blockpi.network/v1/rpc/public",
    "pocket": "https://arb-one.api.pocket.network",
    "meowrpc": "https://arbitrum.meowrpc.com",
    "arbitrum": "https://arb1.arbitrum.io/rpc",
    "alchemy": "https://arb-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}",
    "dwellir": "https://api-arbitrum-mainnet-archive.n.dwellir.com/2ccf18bf-2916-4198-8856-42172854353c",
    "poolz": "https://rpc.poolz.finance/arbitrum",
    "onfinality": "https://arbitrum.api.onfinality.io/public",
    "therpc": "https://arbitrum.therpc.io",
    "omniatech": "https://endpoints.omniatech.io/v1/arbitrum/one/public",
    "callstaticrpc_ws": "wss://arbitrum.callstaticrpc.com",
    "stateless": "https://api.stateless.solutions/arbitrum-one/v1/demo",
    "stackup": "https://public.stackup.sh/api/v1/node/arbitrum-one",
    "gatewayfm": "https://rpc.arb1.arbitrum.gateway.fm",
    "unifra": "https://arb-mainnet-public.unifra.io",
    "alchemy_demo": "https://arb-mainnet.g.alchemy.com/v2/demo",
}
CAMELOT_V2_SUBGRAPH = os.getenv(
    "CAMELOT_V2_SUBGRAPH",
    "https://gateway.thegraph.com/api/subgraphs/id/8zagLSufxk5cVhzkzai3tyABwJh53zxn9tmUYJcJxijG",
)
UNISWAP_V2_SUBGRAPH = os.getenv(
    "UNISWAP_V2_SUBGRAPH",
    "https://gateway.thegraph.com/api/subgraphs/id/CStW6CSQbHoXsgKuVCrk3uShGA4JX3CAzzv2x9zaGf8w",
)
SUSHISWAP_V2_SUBGRAPH = os.getenv(
    "SUSHISWAP_V2_SUBGRAPH",
    "https://gateway.thegraph.com/api/subgraphs/id/8yBXBTMfdhsoE5QCf7KnoPmQb7QAWtRzESfYjiCjGEM9",
)
GRAPH_API_KEY = os.getenv("GRAPH_API_KEY")

PAIR_QUERY = """
query Pairs($lastId: String!, $first: Int!) {
  pairs(first: $first, orderBy: id, orderDirection: asc, where: { id_gt: $lastId }) {
    id
    reserve0
    reserve1
    token0 { id symbol decimals }
    token1 { id symbol decimals }
  }
  _meta { block { number } }
}
"""

BALANCE_OF_SIG = "70a08231"
GET_RESERVES_SIG = "0902f1ac"


@dataclass
class TokenInfo:
    address: str
    symbol: str
    decimals: int


@dataclass
class PairInfo:
    address: str
    token0: TokenInfo
    token1: TokenInfo
    reserve0: Decimal
    reserve1: Decimal


def chunked(values: List[PairInfo], size: int) -> Iterable[List[PairInfo]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def init_pairs_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pairs (
          dex TEXT NOT NULL,
          pair_id TEXT NOT NULL,
          token0 TEXT NOT NULL,
          token1 TEXT NOT NULL,
          token0_symbol TEXT,
          token1_symbol TEXT,
          token0_decimals INTEGER,
          token1_decimals INTEGER,
          last_seen_block INTEGER,
          PRIMARY KEY (dex, pair_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scan_state (
          dex TEXT PRIMARY KEY,
          last_pair_id TEXT,
          last_block INTEGER,
          updated_at INTEGER
        )
        """
    )
    return conn


def load_scan_state(conn: sqlite3.Connection, dex: str) -> Tuple[str, Optional[int]]:
    row = conn.execute(
        "SELECT last_pair_id, last_block FROM scan_state WHERE dex = ?",
        (dex,),
    ).fetchone()
    if not row:
        return "", None
    return row[0] or "", row[1]


def upsert_pairs(
    conn: sqlite3.Connection,
    dex: str,
    pairs: List[PairInfo],
    block_number: Optional[int],
    last_pair_id: str,
) -> None:
    now = int(time.time())
    conn.executemany(
        """
        INSERT INTO pairs (
          dex, pair_id, token0, token1, token0_symbol, token1_symbol,
          token0_decimals, token1_decimals, last_seen_block
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dex, pair_id) DO UPDATE SET
          token0_symbol=excluded.token0_symbol,
          token1_symbol=excluded.token1_symbol,
          token0_decimals=excluded.token0_decimals,
          token1_decimals=excluded.token1_decimals,
          last_seen_block=excluded.last_seen_block
        """,
        [
            (
                dex,
                pair.address,
                pair.token0.address,
                pair.token1.address,
                pair.token0.symbol,
                pair.token1.symbol,
                pair.token0.decimals,
                pair.token1.decimals,
                block_number,
            )
            for pair in pairs
        ],
    )
    conn.execute(
        """
        INSERT INTO scan_state (dex, last_pair_id, last_block, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(dex) DO UPDATE SET
          last_pair_id=excluded.last_pair_id,
          last_block=excluded.last_block,
          updated_at=excluded.updated_at
        """,
        (dex, last_pair_id, block_number, now),
    )
    conn.commit()


def gql_post(url: str, query: str, variables: Dict[str, object]) -> Dict[str, object]:
    for attempt in range(4):
        try:
            headers: Dict[str, str] = {"Content-Type": "application/json"}
            if "gateway.thegraph.com" in url:
                if not GRAPH_API_KEY:
                    raise RuntimeError("GRAPH_API_KEY is required for gateway.thegraph.com")
                headers["Authorization"] = f"Bearer {GRAPH_API_KEY}"
            with httpx.Client(timeout=20) as client:
                resp = client.post(
                    url, json={"query": query, "variables": variables}, headers=headers
                )
            resp.raise_for_status()
            payload = resp.json()
            if "errors" in payload:
                raise RuntimeError(payload["errors"])
            return payload["data"]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.25 * (attempt + 1))
    raise RuntimeError("GraphQL request failed after retries")


def normalize_rpc_url(url: str) -> str:
    alchemy_key = os.getenv("ALCHEMY_API_KEY")
    if "${ALCHEMY_API_KEY}" in url:
        if not alchemy_key:
            return ""
        return url.replace("${ALCHEMY_API_KEY}", alchemy_key)
    return url


def rpc_call(url: str, method: str, params: List[object]) -> str:
    for attempt in range(4):
        try:
            with httpx.Client(timeout=20) as client:
                resp = client.post(
                    normalize_rpc_url(url),
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": method,
                        "params": params,
                    },
                )
            resp.raise_for_status()
            payload = resp.json()
            if "error" in payload:
                raise RuntimeError(payload["error"])
            return payload["result"]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.25 * (attempt + 1))
    raise RuntimeError("RPC request failed after retries")


def encode_call(sig: str, args: List[str]) -> str:
    data = sig
    for arg in args:
        data += arg.rjust(64, "0")
    return "0x" + data


def addr_to_arg(addr: str) -> str:
    return addr.lower().replace("0x", "")


def fetch_pairs_page(
    subgraph: str,
    last_id: str,
    first: int,
) -> Tuple[List[PairInfo], Optional[int]]:
    data = gql_post(
        subgraph,
        PAIR_QUERY,
        {"lastId": last_id, "first": first},
    )
    batch = data.get("pairs", [])
    block_number = None
    if "_meta" in data:
        meta = data.get("_meta") or {}
        block = meta.get("block") or {}
        if "number" in block:
            block_number = int(block["number"])
    pairs: List[PairInfo] = []
    for row in batch:
        token0 = TokenInfo(
            address=row["token0"]["id"],
            symbol=row["token0"]["symbol"],
            decimals=int(row["token0"]["decimals"]),
        )
        token1 = TokenInfo(
            address=row["token1"]["id"],
            symbol=row["token1"]["symbol"],
            decimals=int(row["token1"]["decimals"]),
        )
        pairs.append(
            PairInfo(
                address=row["id"],
                token0=token0,
                token1=token1,
                reserve0=Decimal(row["reserve0"]),
                reserve1=Decimal(row["reserve1"]),
            )
        )
    return pairs, block_number


def fetch_pairs(subgraph: str, max_pairs: int, page_size: int = 200) -> List[PairInfo]:
    pairs: List[PairInfo] = []
    last_id = ""
    limit = max_pairs if max_pairs > 0 else None
    while limit is None or len(pairs) < limit:
        fetch_size = page_size
        if limit is not None:
            fetch_size = min(page_size, limit - len(pairs))
        batch, _ = fetch_pairs_page(subgraph, last_id, fetch_size)
        if not batch:
            break
        pairs.extend(batch)
        last_id = batch[-1].address
    return pairs


def decode_uint256(hexdata: str) -> int:
    if hexdata.startswith("0x"):
        hexdata = hexdata[2:]
    if not hexdata:
        return 0
    return int(hexdata, 16)


def fetch_balance(rpc_url: str, token: str, owner: str) -> int:
    data = encode_call(BALANCE_OF_SIG, [addr_to_arg(owner)])
    result = rpc_call(rpc_url, "eth_call", [{"to": token, "data": data}, "latest"])
    return decode_uint256(result)


def fetch_reserves_raw(rpc_url: str, pair: str) -> Tuple[int, int]:
    data = encode_call(GET_RESERVES_SIG, [])
    result = rpc_call(rpc_url, "eth_call", [{"to": pair, "data": data}, "latest"])
    raw = result[2:]
    if len(raw) < 128:
        raise RuntimeError("Unexpected getReserves response")
    reserve0 = int(raw[0:64], 16)
    reserve1 = int(raw[64:128], 16)
    return reserve0, reserve1


def format_amount(raw: int, decimals: int) -> Decimal:
    return Decimal(raw) / (Decimal(10) ** decimals)


def build_rpc_pool(rpc_urls: Optional[str]) -> List[str]:
    if rpc_urls:
        return [normalize_rpc_url(url.strip()) for url in rpc_urls.split(",") if url.strip()]
    return [normalize_rpc_url(ARBITRUM_RPC_URL)]


def scan_pairs(
    rpc_urls: List[str],
    pairs: List[PairInfo],
    min_imbalance: Decimal,
    rotate_rpc: bool,
) -> List[str]:
    results: List[str] = []
    rpc_count = max(len(rpc_urls), 1)
    for pair in pairs:
        last_exc: Optional[Exception] = None
        for attempt in range(rpc_count):
            rpc_url = rpc_urls[(hash(pair.address) + attempt) % rpc_count] if rotate_rpc else rpc_urls[0]
            try:
                reserve0_raw, reserve1_raw = fetch_reserves_raw(rpc_url, pair.address)
                balance0_raw = fetch_balance(rpc_url, pair.token0.address, pair.address)
                balance1_raw = fetch_balance(rpc_url, pair.token1.address, pair.address)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if not rotate_rpc:
                    break
        if last_exc is not None:
            print(f"skip {pair.address} ({last_exc})", file=sys.stderr)
            continue

        extra0 = balance0_raw - reserve0_raw
        extra1 = balance1_raw - reserve1_raw

        extra0_amt = format_amount(extra0, pair.token0.decimals)
        extra1_amt = format_amount(extra1, pair.token1.decimals)

        if extra0_amt >= min_imbalance or extra1_amt >= min_imbalance:
            results.append(
                "pair={addr} token0={sym0} extra0={extra0} token1={sym1} extra1={extra1}".format(
                    addr=pair.address,
                    sym0=pair.token0.symbol,
                    extra0=extra0_amt,
                    sym1=pair.token1.symbol,
                    extra1=extra1_amt,
                )
            )
    return results


def build_pairs(dex: str, max_pairs: int) -> List[PairInfo]:
    pairs: List[PairInfo] = []
    if dex in ("camelot", "both", "all"):
        pairs.extend(fetch_pairs(CAMELOT_V2_SUBGRAPH, max_pairs))
    if dex in ("uniswapv2", "both", "all"):
        pairs.extend(fetch_pairs(UNISWAP_V2_SUBGRAPH, max_pairs))
    if dex in ("sushiswapv2", "sushiswap", "all"):
        pairs.extend(fetch_pairs(SUSHISWAP_V2_SUBGRAPH, max_pairs))
    return pairs


def load_pairs_from_db(
    conn: sqlite3.Connection, dex: str, limit: int
) -> List[PairInfo]:
    sql = """
        SELECT pair_id, token0, token1, token0_symbol, token1_symbol, token0_decimals, token1_decimals
        FROM pairs
        WHERE dex = ?
        ORDER BY pair_id ASC
    """
    params = [dex]
    if limit > 0:
        sql += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    pairs: List[PairInfo] = []
    for row in rows:
        pairs.append(
            PairInfo(
                address=row[0],
                token0=TokenInfo(
                    address=row[1],
                    symbol=row[3] or "UNKNOWN",
                    decimals=int(row[5] or 18),
                ),
                token1=TokenInfo(
                    address=row[2],
                    symbol=row[4] or "UNKNOWN",
                    decimals=int(row[6] or 18),
                ),
                reserve0=Decimal(0),
                reserve1=Decimal(0),
            )
        )
    return pairs


def update_pair_scan_state(
    conn: sqlite3.Connection, dex: str, pair_ids: List[str], block_number: Optional[int]
) -> None:
    if block_number is None or not pair_ids:
        return
    conn.executemany(
        """
        UPDATE pairs SET last_seen_block = ?
        WHERE dex = ? AND pair_id = ?
        """,
        [(block_number, dex, pair_id) for pair_id in pair_ids],
    )
    conn.commit()


def crawl_pairs_to_db(
    dex: str,
    subgraph: str,
    conn: sqlite3.Connection,
    batch_size: int,
    max_pairs: int,
    resume: bool,
) -> int:
    last_id = ""
    if resume:
        last_id, _ = load_scan_state(conn, dex)
    total = 0
    limit = max_pairs if max_pairs > 0 else None
    while True:
        if limit is not None and total >= limit:
            break
        fetch_size = batch_size
        if limit is not None:
            fetch_size = min(batch_size, limit - total)
        batch, block_number = fetch_pairs_page(subgraph, last_id, fetch_size)
        if not batch:
            break
        last_id = batch[-1].address
        upsert_pairs(conn, dex, batch, block_number, last_id)
        total += len(batch)
        print(f"dex={dex} batch={len(batch)} total={total} last_id={last_id}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal v2 pair skim scanner for Arbitrum (Camelot + Uniswap v2 + SushiSwap)."
    )
    parser.add_argument(
        "--dex",
        choices=["camelot", "uniswapv2", "sushiswap", "sushiswapv2", "both", "all"],
        default="all",
        help="Which v2 subgraph to scan.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=200,
        help="Max pairs per subgraph to scan.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for subgraph crawling.",
    )
    parser.add_argument(
        "--db-path",
        default="skim_pairs.db",
        help="SQLite db path for pair crawl state.",
    )
    parser.add_argument(
        "--crawl",
        action="store_true",
        help="Crawl pairs into the sqlite db and exit.",
    )
    parser.add_argument(
        "--scan-db",
        action="store_true",
        help="Scan pairs from sqlite db instead of subgraph.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume crawling from the last stored pair id.",
    )
    parser.add_argument(
        "--min-imbalance",
        type=Decimal,
        default=Decimal("0.01"),
        help="Minimum extra token amount to flag.",
    )
    parser.add_argument(
        "--rpc-url",
        default=ARBITRUM_RPC_URL,
        help="Arbitrum RPC URL (default: public RPC).",
    )
    parser.add_argument(
        "--rpc-urls",
        default=os.getenv("ARBITRUM_RPC_URLS", ""),
        help="Comma-separated RPC URLs to rotate across.",
    )
    parser.add_argument(
        "--rotate-rpc",
        action="store_true",
        help="Rotate across RPC URLs when set.",
    )
    args = parser.parse_args()

    if args.crawl:
        conn = init_pairs_db(args.db_path)
        if args.dex in ("camelot", "both", "all"):
            crawl_pairs_to_db(
                "camelot",
                CAMELOT_V2_SUBGRAPH,
                conn,
                args.batch_size,
                args.max_pairs,
                args.resume,
            )
        if args.dex in ("uniswapv2", "both", "all"):
            crawl_pairs_to_db(
                "uniswapv2",
                UNISWAP_V2_SUBGRAPH,
                conn,
                args.batch_size,
                args.max_pairs,
                args.resume,
            )
        if args.dex in ("sushiswapv2", "sushiswap", "all"):
            crawl_pairs_to_db(
                "sushiswapv2",
                SUSHISWAP_V2_SUBGRAPH,
                conn,
                args.batch_size,
                args.max_pairs,
                args.resume,
            )
        return

    if args.scan_db:
        conn = init_pairs_db(args.db_path)
        pairs: List[PairInfo] = []
        if args.dex in ("camelot", "both", "all"):
            pairs.extend(load_pairs_from_db(conn, "camelot", args.max_pairs))
        if args.dex in ("uniswapv2", "both", "all"):
            pairs.extend(load_pairs_from_db(conn, "uniswapv2", args.max_pairs))
        if args.dex in ("sushiswapv2", "sushiswap", "all"):
            pairs.extend(load_pairs_from_db(conn, "sushiswapv2", args.max_pairs))
    else:
        pairs = build_pairs(args.dex, args.max_pairs)
    if not pairs:
        print("no pairs loaded")
        return

    rpc_urls = build_rpc_pool(args.rpc_urls if args.rpc_urls else args.rpc_url)
    hits = scan_pairs(rpc_urls, pairs, args.min_imbalance, args.rotate_rpc)
    if not hits:
        print("no skim opportunities found")
        return

    for line in hits:
        print(line)


if __name__ == "__main__":
    main()

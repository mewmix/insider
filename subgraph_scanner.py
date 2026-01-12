import argparse
import os
import sqlite3
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
from dotenv import load_dotenv


ORDERBOOK_SUBGRAPH = os.getenv(
    "ORDERBOOK_SUBGRAPH",
    "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn",
)
POSITIONS_SUBGRAPH = os.getenv(
    "POSITIONS_SUBGRAPH",
    "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/positions-subgraph/0.0.7/gn",
)
ACTIVITY_SUBGRAPH = os.getenv(
    "ACTIVITY_SUBGRAPH",
    "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/activity-subgraph/0.0.4/gn",
)
OI_SUBGRAPH = os.getenv(
    "OI_SUBGRAPH",
    "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/oi-subgraph/0.0.6/gn",
)
PNL_SUBGRAPH = os.getenv(
    "PNL_SUBGRAPH",
    "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/pnl-subgraph/0.0.14/gn",
)

USDC_DECIMALS = 1e6


ORDER_FILLED_QUERY = """
query OrderFilledEvents($addr: String!, $lastId: ID!, $first: Int!, $since: BigInt) {
  orderFilledEvents(
    first: $first,
    orderBy: id,
    orderDirection: asc,
    where: {
      or: [
        { id_gt: $lastId, maker: $addr, timestamp_gte: $since },
        { id_gt: $lastId, taker: $addr, timestamp_gte: $since }
      ]
    }
  ) {
    id
    transactionHash
    timestamp
    maker
    taker
    makerAssetId
    takerAssetId
    makerAmountFilled
    takerAmountFilled
    fee
  }
}
"""

TOKEN_CONDITIONS_QUERY = """
query TokenIdConditions($ids: [String!]) {
  tokenIdConditions(where: { id_in: $ids }) {
    id
    outcomeIndex
    complement
    condition {
      id
      payouts
    }
  }
}
"""


USER_POSITIONS_QUERY = """
query UserPositions($user: String!) {
  userPositions(where: { user: $user }) {
    id
    user
    tokenId
    amount
    avgPrice
    realizedPnl
    totalBought
  }
}
"""

REDEMPTIONS_QUERY = """
query Redemptions($redeemer: String!) {
  redemptions(where: { redeemer: $redeemer }) {
    id
    timestamp
    redeemer
    condition
    payout
  }
}
"""

OPEN_INTEREST_QUERY = """
query MarketOpenInterests($ids: [ID!]) {
  marketOpenInterests(where: { id_in: $ids }) {
    id
    amount
  }
}
"""


def chunked(values: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(values), size):
        yield list(values[i : i + size])


def gql_post(url: str, query: str, variables: Dict[str, object]) -> Dict[str, object]:
    for attempt in range(5):
        try:
            with httpx.Client(timeout=20) as client:
                resp = client.post(url, json={"query": query, "variables": variables})
            resp.raise_for_status()
            payload = resp.json()
            if "errors" in payload:
                raise RuntimeError(payload["errors"])
            return payload["data"]
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("GraphQL request failed after retries")


def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS order_fills (
          id TEXT PRIMARY KEY,
          tx_hash TEXT NOT NULL,
          timestamp INTEGER NOT NULL,
          maker TEXT NOT NULL,
          taker TEXT NOT NULL,
          maker_asset_id TEXT NOT NULL,
          taker_asset_id TEXT NOT NULL,
          maker_amount_filled TEXT NOT NULL,
          taker_amount_filled TEXT NOT NULL,
          fee TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS token_conditions (
          token_id TEXT PRIMARY KEY,
          condition_id TEXT NOT NULL,
          outcome_index INTEGER NOT NULL,
          complement TEXT NOT NULL,
          payouts TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_positions (
          id TEXT PRIMARY KEY,
          user TEXT NOT NULL,
          token_id TEXT NOT NULL,
          amount TEXT NOT NULL,
          avg_price TEXT NOT NULL,
          realized_pnl TEXT NOT NULL,
          total_bought TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS redemptions (
          id TEXT PRIMARY KEY,
          timestamp INTEGER NOT NULL,
          redeemer TEXT NOT NULL,
          condition_id TEXT NOT NULL,
          payout TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS open_interest (
          condition_id TEXT PRIMARY KEY,
          amount TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scan_state (
          k TEXT PRIMARY KEY,
          v TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def get_state(conn: sqlite3.Connection, key: str, default: str) -> str:
    row = conn.execute("SELECT v FROM scan_state WHERE k = ?", (key,)).fetchone()
    return row[0] if row else default


def set_state(conn: sqlite3.Connection, key: str, val: str) -> None:
    conn.execute(
        "INSERT INTO scan_state(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        (key, val),
    )


def fetch_order_fills(
    address: str,
    last_id: str,
    page_size: int,
    since_ts: Optional[int],
) -> List[Dict[str, object]]:
    data = gql_post(
        ORDERBOOK_SUBGRAPH,
        ORDER_FILLED_QUERY,
        {"addr": address, "lastId": last_id, "first": page_size, "since": since_ts or 0},
    )
    return data["orderFilledEvents"]


def store_order_fills(conn: sqlite3.Connection, events: List[Dict[str, object]]) -> None:
    conn.executemany(
        """
        INSERT OR IGNORE INTO order_fills (
          id, tx_hash, timestamp, maker, taker, maker_asset_id, taker_asset_id,
          maker_amount_filled, taker_amount_filled, fee
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                e["id"],
                e["transactionHash"],
                int(e["timestamp"]),
                e["maker"],
                e["taker"],
                e["makerAssetId"],
                e["takerAssetId"],
                e["makerAmountFilled"],
                e["takerAmountFilled"],
                e["fee"],
            )
            for e in events
        ],
    )


def fetch_token_conditions(token_ids: Sequence[str]) -> List[Dict[str, object]]:
    data = gql_post(POSITIONS_SUBGRAPH, TOKEN_CONDITIONS_QUERY, {"ids": token_ids})
    return data["tokenIdConditions"]


def store_token_conditions(conn: sqlite3.Connection, rows: List[Dict[str, object]]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO token_conditions (
          token_id, condition_id, outcome_index, complement, payouts
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                r["id"],
                r["condition"]["id"],
                int(r["outcomeIndex"]),
                r["complement"],
                ",".join(map(str, r["condition"].get("payouts", []) or [])),
            )
            for r in rows
        ],
    )


def fetch_user_positions(user: str) -> List[Dict[str, object]]:
    data = gql_post(PNL_SUBGRAPH, USER_POSITIONS_QUERY, {"user": user})
    return data["userPositions"]


def store_user_positions(conn: sqlite3.Connection, positions: List[Dict[str, object]]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO user_positions (
          id, user, token_id, amount, avg_price, realized_pnl, total_bought
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                p["id"],
                p["user"],
                p["tokenId"],
                p["amount"],
                p["avgPrice"],
                p["realizedPnl"],
                p["totalBought"],
            )
            for p in positions
        ],
    )


def summarize_conditions(conn: sqlite3.Connection) -> List[Tuple[str, int, int]]:
    return conn.execute(
        """
        WITH tokens AS (
          SELECT maker_asset_id AS token_id FROM order_fills
          UNION ALL
          SELECT taker_asset_id AS token_id FROM order_fills
        )
        SELECT tc.condition_id, tc.outcome_index, COUNT(*) as fills
        FROM tokens t
        JOIN token_conditions tc ON tc.token_id = t.token_id
        GROUP BY tc.condition_id, tc.outcome_index
        ORDER BY fills DESC
        """
    ).fetchall()


def get_condition_ids(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT DISTINCT condition_id FROM token_conditions").fetchall()
    return [r[0] for r in rows if r[0]]


def fetch_redemptions(redeemer: str) -> List[Dict[str, object]]:
    data = gql_post(ACTIVITY_SUBGRAPH, REDEMPTIONS_QUERY, {"redeemer": redeemer})
    return data["redemptions"]


def store_redemptions(conn: sqlite3.Connection, redemptions: List[Dict[str, object]]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO redemptions (
          id, timestamp, redeemer, condition_id, payout
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                r["id"],
                int(r["timestamp"]),
                r["redeemer"],
                r["condition"],
                r["payout"],
            )
            for r in redemptions
        ],
    )


def fetch_open_interest(condition_ids: Sequence[str]) -> List[Dict[str, object]]:
    data = gql_post(OI_SUBGRAPH, OPEN_INTEREST_QUERY, {"ids": condition_ids})
    return data["marketOpenInterests"]


def store_open_interest(conn: sqlite3.Connection, rows: List[Dict[str, object]]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO open_interest (
          condition_id, amount
        ) VALUES (?, ?)
        """,
        [(r["id"], r["amount"]) for r in rows],
    )


def summarize_pnl(conn: sqlite3.Connection) -> float:
    row = conn.execute("SELECT SUM(CAST(realized_pnl AS REAL)) FROM user_positions").fetchone()
    return (row[0] or 0.0) / USDC_DECIMALS


def summarize_redemptions(conn: sqlite3.Connection) -> Tuple[int, float]:
    row = conn.execute("SELECT COUNT(*), SUM(CAST(payout AS REAL)) FROM redemptions").fetchone()
    return (row[0] or 0, (row[1] or 0.0) / USDC_DECIMALS)


def summarize_open_interest(conn: sqlite3.Connection) -> float:
    row = conn.execute("SELECT SUM(CAST(amount AS REAL)) FROM open_interest").fetchone()
    return (row[0] or 0.0) / USDC_DECIMALS


def run_once(
    conn: sqlite3.Connection,
    address: str,
    page_size: int,
    since_ts: Optional[int],
) -> int:
    last_id = get_state(conn, "last_order_filled_id", "0")
    total = 0
    new_token_ids: List[str] = []
    
    # Fetch user positions for PNL
    positions = fetch_user_positions(address)
    store_user_positions(conn, positions)

    # Fetch redemptions
    redemptions = fetch_redemptions(address)
    store_redemptions(conn, redemptions)

    while True:
        events = fetch_order_fills(address, last_id, page_size, since_ts)
        if not events:
            break
        store_order_fills(conn, events)
        total += len(events)
        last_id = events[-1]["id"]
        for e in events:
            new_token_ids.append(e["makerAssetId"])
            new_token_ids.append(e["takerAssetId"])
        set_state(conn, "last_order_filled_id", str(last_id))
        conn.commit()

    unique_ids = sorted(set(new_token_ids))
    if unique_ids:
        for batch in chunked(unique_ids, 200):
            rows = fetch_token_conditions(batch)
            store_token_conditions(conn, rows)
        conn.commit()

    condition_ids = get_condition_ids(conn)
    if condition_ids:
        for batch in chunked(condition_ids, 200):
            rows = fetch_open_interest(batch)
            store_open_interest(conn, rows)
        conn.commit()
    return total


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Polymarket subgraph scanner")
    parser.add_argument("--address", required=True, help="Wallet address to scan")
    parser.add_argument("--db", default="subgraph_scanner.db", help="SQLite path")
    parser.add_argument("--page-size", type=int, default=200, help="GraphQL page size")
    parser.add_argument("--since-ts", type=int, default=None, help="Unix timestamp filter")
    parser.add_argument("--follow", action="store_true", help="Poll continuously")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Polling interval")
    args = parser.parse_args()

    address = args.address.lower()
    conn = init_db(args.db)
    try:
        while True:
            new_events = run_once(conn, address, args.page_size, args.since_ts)
            conditions = summarize_conditions(conn)
            total_pnl = summarize_pnl(conn)
            red_count, red_total = summarize_redemptions(conn)
            total_oi = summarize_open_interest(conn)
            print(
                "new_fills={fills} conditions={conditions} pnl=${pnl:,.2f} "
                "redemptions={red_count} (${red_total:,.2f}) oi=${oi:,.2f}".format(
                    fills=new_events,
                    conditions=len(conditions),
                    pnl=total_pnl,
                    red_count=red_count,
                    red_total=red_total,
                    oi=total_oi,
                )
            )
            for condition_id, outcome_index, fills in conditions[:10]:
                print(f"{condition_id} outcome={outcome_index} fills={fills}")
            if not args.follow:
                break
            time.sleep(args.poll_seconds)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

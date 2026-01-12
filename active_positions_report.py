import argparse
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Sequence, Tuple

import httpx


ORDERBOOK_SUBGRAPH = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"
POSITIONS_SUBGRAPH = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/positions-subgraph/0.0.7/gn"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"

DEFAULT_TIMEOUT = 30.0

USDC_DECIMALS = 1e6


ORDER_FILLED_QUERY = """
query OrderFilledEvents($since: BigInt!, $lastId: ID!, $first: Int!) {
  orderFilledEvents(
    first: $first,
    orderBy: id,
    orderDirection: asc,
    where: { id_gt: $lastId, timestamp_gte: $since }
  ) {
    id
    timestamp
    maker
    taker
  }
}
"""


USER_BALANCES_QUERY = """
query UserBalances($user: String!) {
  userBalances(where: { user: $user }) {
    balance
    asset {
      id
      outcomeIndex
      condition {
        id
      }
    }
  }
}
"""


def chunked(values: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(values), size):
        yield list(values[i : i + size])


def gql_post(
    url: str,
    query: str,
    variables: Dict[str, object],
    timeout: float,
) -> Dict[str, object]:
    for attempt in range(5):
        try:
            with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
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


def fetch_active_addresses(
    since_ts: int,
    page_size: int,
    max_events: int,
    timeout: float,
) -> List[str]:
    last_id = ""
    seen = set()
    fetched = 0
    while fetched < max_events:
        data = gql_post(
            ORDERBOOK_SUBGRAPH,
            ORDER_FILLED_QUERY,
            {"since": since_ts, "lastId": last_id, "first": page_size},
            timeout,
        )
        events = data["orderFilledEvents"]
        if not events:
            break
        for event in events:
            maker = str(event.get("maker", "")).lower()
            taker = str(event.get("taker", "")).lower()
            if maker:
                seen.add(maker)
            if taker:
                seen.add(taker)
        last_id = events[-1]["id"]
        fetched += len(events)
        if len(events) < page_size:
            break
    return sorted(seen)


def fetch_user_balances(user: str, timeout: float) -> List[Dict[str, object]]:
    data = gql_post(POSITIONS_SUBGRAPH, USER_BALANCES_QUERY, {"user": user}, timeout)
    return data["userBalances"]


def fetch_market_metadata(condition_id: str, timeout: float) -> Dict[str, str]:
    params = {"conditionId": condition_id}
    with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
        resp = client.get(f"{GAMMA_API_BASE}/markets", params=params)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return {}
    market = data[0]
    return {
        "title": market.get("question") or market.get("title") or "",
        "slug": market.get("slug") or "",
    }


def parse_balance(raw: str) -> float:
    try:
        return float(raw) / USDC_DECIMALS
    except ValueError:
        return 0.0


def summarize_positions(
    user_positions: Dict[str, List[Dict[str, object]]]
) -> Tuple[Dict[str, float], Dict[str, Dict[int, float]]]:
    total_by_user: Dict[str, float] = {}
    by_condition: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for user, balances in user_positions.items():
        total = 0.0
        for b in balances:
            balance_raw = str(b.get("balance", "0") or "0")
            amount = abs(parse_balance(balance_raw))
            if amount == 0:
                continue
            asset = b.get("asset") or {}
            outcome_index = int(asset.get("outcomeIndex", 0))
            condition = asset.get("condition") or {}
            condition_id = str(condition.get("id", "")).lower()
            if condition_id:
                by_condition[condition_id][outcome_index] += amount
            total += amount
        total_by_user[user] = total
    return total_by_user, by_condition


def utc_day_start(ts: int) -> int:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return int(datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp())


def main() -> None:
    parser = argparse.ArgumentParser(description="Active positions overlap report")
    parser.add_argument("--since-ts", type=int, default=None, help="Unix timestamp start (UTC)")
    parser.add_argument("--max-traders", type=int, default=20, help="Top traders to report")
    parser.add_argument("--max-candidates", type=int, default=200, help="Active traders to scan")
    parser.add_argument("--page-size", type=int, default=200, help="GraphQL page size")
    parser.add_argument("--max-events", type=int, default=2000, help="Max events to scan")
    parser.add_argument("--contrary-min-share", type=float, default=0.2, help="Min share per outcome to mark contrary")
    parser.add_argument("--with-metadata", action="store_true", help="Fetch Gamma titles for top markets")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout seconds")
    args = parser.parse_args()

    now_ts = int(time.time())
    since_ts = args.since_ts if args.since_ts is not None else utc_day_start(now_ts)

    active_addresses = fetch_active_addresses(
        since_ts,
        args.page_size,
        args.max_events,
        args.timeout,
    )
    if not active_addresses:
        print("No active traders found for today.")
        return
    candidates = active_addresses[: args.max_candidates]

    user_positions: Dict[str, List[Dict[str, object]]] = {}
    for user in candidates:
        balances = fetch_user_balances(user, args.timeout)
        user_positions[user] = balances

    total_by_user, by_condition = summarize_positions(user_positions)
    ranked = sorted(total_by_user.items(), key=lambda x: x[1], reverse=True)
    top_traders = [addr for addr, _ in ranked[: args.max_traders]]

    print(
        f"Active since UTC {datetime.fromtimestamp(since_ts, tz=timezone.utc).isoformat()} "
        f"candidates={len(candidates)}"
    )
    print(f"{'Rank':<5} {'Address':<45} {'Pos Size (shares)':>20} {'Positions':>10}")
    print("-" * 90)
    for i, addr in enumerate(top_traders, start=1):
        balances = user_positions.get(addr, [])
        pos_count = sum(1 for b in balances if parse_balance(str(b.get("balance", "0"))) != 0)
        total = total_by_user.get(addr, 0.0)
        print(f"{i:<5} {addr:<45} {total:>20,.2f} {pos_count:>10}")

    top_conditions = sorted(
        by_condition.items(),
        key=lambda x: sum(x[1].values()),
        reverse=True,
    )[:20]

    metadata_cache: Dict[str, Dict[str, str]] = {}
    if args.with_metadata:
        for condition_id, _ in top_conditions:
            try:
                metadata_cache[condition_id] = fetch_market_metadata(condition_id, args.timeout)
            except Exception:
                metadata_cache[condition_id] = {}

    print("\nOverlap/Contrary by Market")
    print(f"{'Condition ID':<66} {'Total':>12}  Outcomes")
    print("-" * 110)
    for condition_id, outcomes in top_conditions:
        total = sum(outcomes.values())
        outcome_parts = []
        for outcome_index, amount in sorted(outcomes.items()):
            share = amount / total if total else 0.0
            outcome_parts.append(f"{outcome_index}:{amount:,.2f} ({share:.0%})")
        label = condition_id
        if args.with_metadata:
            meta = metadata_cache.get(condition_id) or {}
            title = meta.get("title") or ""
            if title:
                label = f"{condition_id} | {title[:60]}"
        print(f"{label:<66} {total:>12,.2f}  " + " | ".join(outcome_parts))

    print("\nContrary Markets (both sides meaningful)")
    print(f"{'Condition ID':<66} {'Total':>12}  Outcomes")
    print("-" * 110)
    for condition_id, outcomes in top_conditions:
        total = sum(outcomes.values())
        if total == 0:
            continue
        shares = [amount / total for amount in outcomes.values()]
        if len(shares) < 2:
            continue
        if min(shares) < args.contrary_min_share:
            continue
        outcome_parts = []
        for outcome_index, amount in sorted(outcomes.items()):
            share = amount / total if total else 0.0
            outcome_parts.append(f"{outcome_index}:{amount:,.2f} ({share:.0%})")
        label = condition_id
        if args.with_metadata:
            meta = metadata_cache.get(condition_id) or {}
            title = meta.get("title") or ""
            if title:
                label = f"{condition_id} | {title[:60]}"
        print(f"{label:<66} {total:>12,.2f}  " + " | ".join(outcome_parts))


if __name__ == "__main__":
    main()

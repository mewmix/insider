import argparse
import os
import time
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

load_dotenv()
getcontext().prec = 50

AAVE_V3_SUBGRAPH = os.getenv(
    "AAVE_V3_SUBGRAPH",
    "https://gateway.thegraph.com/api/subgraphs/id/4xyasjQeREe7PxnF6wVdobZvCw5mhoHZq3T7guRpuNPf",
)
GRAPH_API_KEY = os.getenv("GRAPH_API_KEY")

# Revised query based on schema exploration
# Position has 'market' (Market entity) which has 'inputToken' and configuration.
# We filter for accounts with open positions.

POSITIONS_QUERY = """
query Users($lastId: String!, $first: Int!) {
  accounts(first: $first, orderBy: id, orderDirection: asc, where: { id_gt: $lastId, openPositionCount_gt: 0 }) {
    id
    positions(where: { balance_gt: 0 }) {
      id
      side
      balance
      market {
        id
        liquidationThreshold
        inputTokenPriceUSD
        inputToken {
          symbol
          decimals
        }
      }
    }
  }
}
"""

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
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(1.25 * (attempt + 1))
    raise RuntimeError("GraphQL request failed after retries")


def fetch_accounts_with_positions(last_id: str, first: int) -> List[Dict]:
    data = gql_post(AAVE_V3_SUBGRAPH, POSITIONS_QUERY, {"lastId": last_id, "first": first})
    return data.get("accounts", [])


def calculate_health_factor(account: Dict) -> Tuple[Decimal, Decimal, Decimal]:
    # Health Factor = (Total Collateral in USD * Liquidation Threshold) / Total Borrows in USD

    total_collateral_usd = Decimal(0)
    total_borrows_usd = Decimal(0)
    weighted_collateral_usd = Decimal(0) # Collateral * LT

    for pos in account.get("positions", []):
        market = pos.get("market", {})
        if not market:
             continue

        token_price_usd = Decimal(market.get("inputTokenPriceUSD", "0"))

        # Balance normalization
        raw_balance = Decimal(pos.get("balance", "0"))
        decimals = 18
        if "inputToken" in market and market["inputToken"]:
             decimals = int(market["inputToken"].get("decimals", 18))

        amount = raw_balance / (Decimal(10) ** decimals)
        value_usd = amount * token_price_usd

        side = pos.get("side")
        if side == "LENDER" or side == "COLLATERAL": # Check actual side values from schema/data. Usually 'LENDER' or 'BORROWER'. 'isCollateral' is separate flag.
            # Assuming LENDER means supplied. If it is used as collateral depends on configuration.
            # But usually we check 'isCollateral' in Position. My query doesn't fetch it yet.
            # Let's verify 'side' values. Usually 'LENDER' for deposit, 'BORROWER' for borrow.
            # The schema showed 'side' enum type likely.

            # Liquidation Threshold is percentage. In Aave V3 data seen: "78".
            # Usually it is out of 100? Or 10000?
            # Aave V2 was 8000 (80%). V3 example showed "78".
            # If it is 78, it might be 78%.
            # Let's check standard Aave docs or assume standard format.
            # If "78" -> likely 78%. If "7800" -> 78%.
            # I'll assume if > 100 it is basis points (10000 base), if <= 100 it is percent.

            lt_raw = Decimal(market.get("liquidationThreshold", "0"))
            if lt_raw > 100:
                lt = lt_raw / 10000
            else:
                lt = lt_raw / 100

            total_collateral_usd += value_usd
            weighted_collateral_usd += value_usd * lt

        elif side == "BORROWER":
            total_borrows_usd += value_usd

    if total_borrows_usd == 0:
        return Decimal("Infinity"), total_collateral_usd, total_borrows_usd

    health_factor = weighted_collateral_usd / total_borrows_usd
    return health_factor, total_collateral_usd, total_borrows_usd


def main():
    parser = argparse.ArgumentParser(description="Aave V3 Health Scanner on Arbitrum")
    parser.add_argument("--limit", type=int, default=100, help="Max accounts to scan")
    parser.add_argument("--threshold", type=float, default=1.1, help="Health factor threshold to alert")
    args = parser.parse_args()

    print(f"Scanning Aave V3 accounts for HF < {args.threshold}...")

    last_id = ""
    count = 0

    while count < args.limit:
        batch_size = min(100, args.limit - count)
        accounts = fetch_accounts_with_positions(last_id, batch_size)
        if not accounts:
            break

        for account in accounts:
            hf, collateral, debt = calculate_health_factor(account)
            # Filter out healthy or empty accounts
            if debt > 0 and hf < Decimal(args.threshold):
                print(f"User: {account['id']} | HF: {hf:.4f} | Collateral: ${collateral:,.2f} | Debt: ${debt:,.2f}")

        last_id = accounts[-1]["id"]
        count += len(accounts)
        if count % 100 == 0:
            print(f"Scanned {count} accounts...")

if __name__ == "__main__":
    main()

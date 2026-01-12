import time
from typing import Dict, List, Sequence, Iterable
import httpx
from collections import defaultdict

PNL_SUBGRAPH = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/pnl-subgraph/0.0.14/gn"
USDC_DECIMALS = 1e6

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
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("GraphQL request failed after retries")

TOP_PNL_QUERY = """
query TopPnl($first: Int!, $skip: Int!) {
  userPositions(
    first: $first, 
    skip: $skip, 
    orderBy: realizedPnl, 
    orderDirection: desc
  ) {
    user
    realizedPnl
  }
}
"""

def fetch_top_positions(limit: int = 1000) -> List[Dict[str, object]]:
    all_positions = []
    page_size = 1000
    skip = 0
    
    # We can fetch up to `limit` or until we run out of high-value positions
    # GraphQL usually limits `skip` + `first` to some value (often 5000 or 6000), 
    # so we should be careful. 
    # Let's try to fetch a few pages. 
    
    while len(all_positions) < limit:
        print(f"Fetching positions... skip={skip}")
        fetch_count = min(page_size, limit - len(all_positions))
        data = gql_post(PNL_SUBGRAPH, TOP_PNL_QUERY, {"first": fetch_count, "skip": skip})
        positions = data["userPositions"]
        if not positions:
            break
        all_positions.extend(positions)
        skip += len(positions)
        
        # Stop if we fetched fewer than requested, meaning we reached the end
        if len(positions) < fetch_count:
            break
            
    return all_positions

def fetch_all_user_positions(user: str) -> List[Dict[str, object]]:

    all_positions = []

    last_id = ""

    while True:

        query = """

        query UserPositions($user: String!, $lastId: ID!) {

          userPositions(first: 1000, where: { user: $user, id_gt: $lastId }, orderBy: id) {

            id

            realizedPnl

          }

        }

        """

        try:

            data = gql_post(PNL_SUBGRAPH, query, {"user": user, "lastId": last_id})

            batch = data["userPositions"]

        except Exception as e:

            print(f"Error fetching for {user}: {e}")

            break

            

        if not batch:

            break

        all_positions.extend(batch)

        last_id = batch[-1]["id"]

        if len(all_positions) > 5000: # Limit to avoid taking forever

            break

    return all_positions



def main():

    print("Fetching top user positions (candidates)...")

    # Fetch more candidates to ensure we find the true net winners

    positions = fetch_top_positions(limit=2000)

    

    candidate_gross_pnl = defaultdict(float)

    for p in positions:

        user = p["user"]

        raw_pnl = float(p["realizedPnl"])

        pnl_usdc = raw_pnl / USDC_DECIMALS

        candidate_gross_pnl[user] += pnl_usdc

        

    sorted_candidates = sorted(candidate_gross_pnl.items(), key=lambda x: x[1], reverse=True)

    top_50_candidates = [user for user, _ in sorted_candidates[:50]]

    

    print(f"\nVerifying Net PnL for top {len(top_50_candidates)} candidates...")

    

    final_leaderboard = []

    

    for i, user in enumerate(top_50_candidates):

        print(f"[{i+1}/{len(top_50_candidates)}] Scanning {user}...")

        user_positions = fetch_all_user_positions(user)

        net_pnl = sum(float(p["realizedPnl"]) for p in user_positions) / USDC_DECIMALS

        final_leaderboard.append((user, net_pnl, len(user_positions)))

        

    final_leaderboard.sort(key=lambda x: x[1], reverse=True)

    

    print("\nTop 20 Accounts by Net Realized PnL:")

    print(f"{'Rank':<5} {'Address':<45} {'Net PnL (USDC)':>20} {'Positions':>10}")

    print("-" * 85)

    

    for i, (user, pnl, count) in enumerate(final_leaderboard[:20]):

        print(f"{i+1:<5} {user:<45} ${pnl:,.2f} {count:>10}")



if __name__ == "__main__":

    main()



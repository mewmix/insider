import argparse
import json
import os
import sqlite3
import time
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from ignore_list import parse_ignore_addresses, is_ignored_address
from policy import parse_allow_addresses, is_allowed_address
from scanner import (
    PairData,
    STABLES,
    WETH,
    build_pair_index,
    fetch_reserves_with_rotation,
    find_best_price_pair,
    load_pairs,
    pair_liquidity_usd,
    swap_out_pair,
    token_price_usd,
)


load_dotenv()


def ternary_search_generic(max_in: Decimal, profit_fn) -> Tuple[Decimal, Decimal]:
    lo = Decimal(0)
    hi = max_in
    for _ in range(60):
        m1 = lo + (hi - lo) / Decimal(3)
        m2 = hi - (hi - lo) / Decimal(3)
        p1 = profit_fn(m1)
        p2 = profit_fn(m2)
        if p1 > p2:
            hi = m2
        else:
            lo = m1
    best_in = (lo + hi) / 2
    best_profit = profit_fn(best_in)
    return best_in, best_profit


def build_adjacency(pairs: List[PairData]) -> Dict[str, List[Tuple[str, PairData]]]:
    adj: Dict[str, List[Tuple[str, PairData]]] = {}
    for pair in pairs:
        t0 = pair.token0.lower()
        t1 = pair.token1.lower()
        adj.setdefault(t0, []).append((t1, pair))
        adj.setdefault(t1, []).append((t0, pair))
    return adj


def select_start_tokens(adj: Dict[str, List[Tuple[str, PairData]]], top_n: int, explicit: List[str]) -> List[str]:
    if explicit:
        return [t.lower() for t in explicit]
    ranked = sorted(adj.keys(), key=lambda k: len(adj[k]), reverse=True)
    return ranked[:top_n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore multi-hop opportunities from DB graph.")
    parser.add_argument("--db-path", default="skim_pairs.db", help="SQLite DB with pairs.")
    parser.add_argument("--dexes", default="uniswapv2,camelot,sushiswapv2")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--rpc-urls", default=os.getenv("ARBITRUM_RPC_URLS", ""))
    parser.add_argument("--start-tokens", default="", help="Comma-separated start tokens.")
    parser.add_argument("--start-top", type=int, default=50, help="Top degree tokens to start from.")
    parser.add_argument("--max-hops", type=int, default=4, help="Max hops per cycle (>=3).")
    parser.add_argument("--max-paths", type=int, default=200, help="Max cycles to output.")
    parser.add_argument("--max-edges-per-token", type=int, default=50, help="Cap edges per token.")
    parser.add_argument("--max-trade-frac", type=Decimal, default=Decimal("0.1"))
    parser.add_argument("--gas-units", type=int, default=500000)
    parser.add_argument("--gas-price-gwei", type=Decimal, default=Decimal("0.02"))
    parser.add_argument("--min-net-profit-usd", type=Decimal, default=Decimal("1.00"))
    parser.add_argument("--min-pair-liquidity-usd", type=Decimal, default=Decimal("500000"))
    parser.add_argument("--safety-bps", type=Decimal, default=Decimal("10"))
    parser.add_argument("--dump", default="multi_hop_opps.jsonl", help="JSONL output file.")
    parser.add_argument("--ignore-addresses", default="", help="Comma-separated addresses to ignore.")
    parser.add_argument("--allow-addresses", default="", help="Comma-separated addresses to allow.")
    args = parser.parse_args()

    dexes = [d.strip() for d in args.dexes.split(",") if d.strip()]
    ignore_addresses = parse_ignore_addresses(args.ignore_addresses)
    allow_addresses = parse_allow_addresses(args.allow_addresses)

    conn = sqlite3.connect(args.db_path)
    pairs: List[PairData] = []
    for dex in dexes:
        pairs.extend(load_pairs(conn, dex, args.max_pairs))

    if ignore_addresses:
        pairs = [
            p
            for p in pairs
            if not (
                is_ignored_address(p.pair_id, ignore_addresses)
                or is_ignored_address(p.token0, ignore_addresses)
                or is_ignored_address(p.token1, ignore_addresses)
            )
        ]
    if allow_addresses:
        pairs = [
            p
            for p in pairs
            if (
                is_allowed_address(p.pair_id, allow_addresses, allow_any=False)
                or (
                    is_allowed_address(p.token0, allow_addresses, allow_any=False)
                    and is_allowed_address(p.token1, allow_addresses, allow_any=False)
                )
            )
        ]

    if not pairs:
        print("no pairs loaded")
        return

    pair_index = build_pair_index(pairs)
    adj = build_adjacency(pairs)
    start_tokens = select_start_tokens(
        adj,
        args.start_top,
        [t.strip() for t in args.start_tokens.split(",") if t.strip()],
    )

    rpc_urls = [u.strip() for u in args.rpc_urls.split(",") if u.strip()]
    if not rpc_urls:
        rpc_urls = [os.getenv("ARBITRUM_RPC_URL", "https://arb1.arbitrum.io/rpc")]

    weth_price, _ = find_best_price_pair(WETH, list(STABLES), pair_index, rpc_urls, 0)

    fee_by_dex = {
        "uniswapv2": Decimal("0.003"),
        "camelot": Decimal("0.005"),
        "sushiswapv2": Decimal("0.005"),
    }

    reserve_cache: Dict[str, Tuple[int, int]] = {}
    results = 0

    def dfs(
        start_token: str,
        current_token: str,
        path_tokens: List[str],
        path_pairs: List[PairData],
    ) -> None:
        nonlocal results
        if results >= args.max_paths:
            return
        if len(path_pairs) >= args.max_hops:
            return

        for (next_token, pair) in adj.get(current_token, [])[: args.max_edges_per_token]:
            if results >= args.max_paths:
                return
            if next_token == start_token and len(path_pairs) >= 2:
                candidate_pairs = path_pairs + [pair]
                candidate_tokens = path_tokens + [start_token]

                try:
                    for p in candidate_pairs:
                        if p.pair_id not in reserve_cache:
                            r0, r1 = fetch_reserves_with_rotation(rpc_urls, p.pair_id, 0)
                            reserve_cache[p.pair_id] = (r0, r1)
                            p.reserve0 = p.reserve0 or Decimal(0)
                            p.reserve1 = p.reserve1 or Decimal(0)
                            p.reserve0 = Decimal(r0) / (Decimal(10) ** p.token0_decimals)
                            p.reserve1 = Decimal(r1) / (Decimal(10) ** p.token1_decimals)

                    if args.min_pair_liquidity_usd > 0:
                        liqs = [
                            pair_liquidity_usd(p, pair_index, rpc_urls, 0, weth_price)
                            for p in candidate_pairs
                        ]
                        if any(l is None or l < args.min_pair_liquidity_usd for l in liqs):
                            continue

                    def profit_fn(amount_in: Decimal) -> Decimal:
                        amt = amount_in
                        for hop_idx, hop_pair in enumerate(candidate_pairs):
                            fee = fee_by_dex.get(hop_pair.dex, Decimal("0.003"))
                            amt = swap_out_pair(amt, candidate_tokens[hop_idx], hop_pair, fee)
                            if amt <= 0:
                                return Decimal(0)
                        return amt - amount_in

                    if candidate_pairs[0].token0.lower() == start_token:
                        max_in = candidate_pairs[0].reserve0 * args.max_trade_frac
                        start_decimals = candidate_pairs[0].token0_decimals
                    else:
                        max_in = candidate_pairs[0].reserve1 * args.max_trade_frac
                        start_decimals = candidate_pairs[0].token1_decimals

                    if max_in <= 0:
                        continue
                    amt_in, profit = ternary_search_generic(max_in, profit_fn)
                    if profit <= 0:
                        continue

                    start_price = token_price_usd(start_token, pair_index, rpc_urls, 0, weth_price)
                    if start_price is None:
                        continue
                    profit_usd = profit * start_price
                    gas_cost_eth = (Decimal(args.gas_units) * args.gas_price_gwei) / Decimal(1e9)
                    if weth_price is None:
                        continue
                    gas_cost_usd = gas_cost_eth * weth_price
                    net_profit_usd = profit_usd - gas_cost_usd
                    if net_profit_usd < args.min_net_profit_usd:
                        continue

                    min_profit_weth_raw = 0
                    if start_token.lower() == WETH.lower() and start_price:
                        min_profit_weth = args.min_net_profit_usd / start_price
                        min_profit_weth_raw = int(min_profit_weth * Decimal(10) ** 18)

                    payload = {
                        "start_token": start_token,
                        "path_tokens": candidate_tokens,
                        "path_pairs": [p.pair_id for p in candidate_pairs],
                        "path_token_decimals": [start_decimals]
                        + [
                            p.token1_decimals if p.token0.lower() == candidate_tokens[idx].lower() else p.token0_decimals
                            for idx, p in enumerate(candidate_pairs)
                        ],
                        "dexes": [p.dex for p in candidate_pairs],
                        "amount_in": str(amt_in),
                        "amount_in_raw": int(amt_in * (Decimal(10) ** start_decimals)),
                        "profit_est": str(profit),
                        "net_profit_usd_est": str(net_profit_usd),
                        "fee_bps": [
                            int(fee_by_dex.get(p.dex, Decimal("0.003")) * Decimal(10000))
                            for p in candidate_pairs
                        ],
                        "safety_bps": int(args.safety_bps),
                        "min_profit_weth_raw": min_profit_weth_raw,
                        "timestamp": int(time.time()),
                    }
                    with open(args.dump, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(payload) + "\n")
                    results += 1
                except Exception:
                    continue
                continue

            if next_token in path_tokens:
                continue
            if ignore_addresses and is_ignored_address(next_token, ignore_addresses):
                continue
            if allow_addresses and not is_allowed_address(next_token, allow_addresses, allow_any=False):
                continue
            if is_ignored_address(pair.pair_id, ignore_addresses):
                continue
            if allow_addresses and not is_allowed_address(pair.pair_id, allow_addresses, allow_any=False):
                continue
            dfs(start_token, next_token, path_tokens + [next_token], path_pairs + [pair])

    for token in start_tokens:
        if results >= args.max_paths:
            break
        if ignore_addresses and is_ignored_address(token, ignore_addresses):
            continue
        if allow_addresses and not is_allowed_address(token, allow_addresses, allow_any=False):
            continue
        dfs(token, token, [token], [])

    print(f"written={results} file={args.dump}")


if __name__ == "__main__":
    main()

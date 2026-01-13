import argparse
import json
import os
import sqlite3
import sys
import time
import asyncio
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Tuple, Set, Any

import httpx
from dotenv import load_dotenv
from web3 import Web3
from web3.exceptions import ContractLogicError
from eth_abi import encode

from skim_scanner import RPC_ENDPOINTS, normalize_rpc_url

load_dotenv()
getcontext().prec = 60

# --- Constants ---

UNISWAP_V2_SUBGRAPH = os.getenv("UNISWAP_V2_SUBGRAPH", "https://gateway.thegraph.com/api/subgraphs/id/CStW6CSQbHoXsgKuVCrk3uShGA4JX3CAzzv2x9zaGf8w")
CAMELOT_V2_SUBGRAPH = os.getenv("CAMELOT_V2_SUBGRAPH", "https://gateway.thegraph.com/api/subgraphs/id/8zagLSufxk5cVhzkzai3tyABwJh53zxn9tmUYJcJxijG")
SUSHISWAP_V2_SUBGRAPH = os.getenv("SUSHISWAP_V2_SUBGRAPH", "https://gateway.thegraph.com/api/subgraphs/id/8yBXBTMfdhsoE5QCf7KnoPmQb7QAWtRzESfYjiCjGEM9")
GRAPH_API_KEY = os.getenv("GRAPH_API_KEY")

WETH = "0x82af49447d8a07e3bd95bd0d56f35241523fbab1"
STABLES = {
    "0xff970a61a04b1ca14834a43f5de4533ebddb5cc8",  # USDC.e
    "0xaf88d065e77c8cc2239327c5edb3a432268e5831",  # USDC
    "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9",  # USDT
    "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1",  # DAI
}
AAVE_SUPPORTED = {
    WETH.lower(),
    "0xaf88d065e77c8cc2239327c5edb3a432268e5831", # USDC
    "0xff970a61a04b1ca14834a43f5de4533ebddb5cc8", # USDC.e
    "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9", # USDT
    "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1", # DAI
    "0x2f2a2543b76a4166549f7aab2e75bef0aefc5b0f", # WBTC
    "0xf97f4df75117a78c1a5a0dbb88af67027ae233c4", # LINK
    "0x912ce59144191c1204e64559fe8253a0e49e6548", # ARB
}

MONSTROSITY_ADDRESS = "0x7e5E849D5a3FBAea7044b4b9e47baBb3d6A60283"
AAVE_V3_POOL_ADDRESS = "0x794a61358D6845594F94dc1DB02A252b5b4814aD"

MONSTROSITY_ABI = [
    {
        "inputs": [
            {
                "components": [
                    {"internalType": "uint8", "name": "action", "type": "uint8"},
                    {"internalType": "address", "name": "target", "type": "address"},
                    {"internalType": "address", "name": "tokenIn", "type": "address"},
                    {"internalType": "address", "name": "tokenOut", "type": "address"},
                    {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                    {"internalType": "uint256", "name": "minAmountOut", "type": "uint256"},
                    {"internalType": "bytes", "name": "extraData", "type": "bytes"}
                ],
                "internalType": "struct Monstrosity.Step[]",
                "name": "steps",
                "type": "tuple[]"
            },
            {"internalType": "uint256", "name": "minProfitUSD", "type": "uint256"}
        ],
        "name": "execute",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

ACTION_V2_SWAP = 1
ACTION_V3_SWAP = 2
ACTION_AAVE_FLASH = 3
ACTION_V2_FLASH_SWAP = 4

@dataclass
class PairData:
    dex: str
    pair_id: str
    token0: str
    token1: str
    token0_symbol: str
    token1_symbol: str
    token0_decimals: int
    token1_decimals: int
    reserve0: Decimal = Decimal(0)
    reserve1: Decimal = Decimal(0)
    fee_bps: int = 30 # Default 30bps

    def other_token(self, token: str) -> str:
        return self.token1 if self.token0.lower() == token.lower() else self.token0

    def get_reserves(self, token_in: str) -> Tuple[Decimal, Decimal]:
        if self.token0.lower() == token_in.lower():
            return self.reserve0, self.reserve1
        return self.reserve1, self.reserve0

@dataclass
class PathStep:
    pair: PairData
    token_in: str
    token_out: str

@dataclass
class ArbitragePath:
    steps: List[PathStep]
    start_token: str
    optimal_amount_in: Decimal = Decimal(0)
    expected_profit: Decimal = Decimal(0)
    flash_source: str = "none" # "aave", "v2_flash", "none"
    flash_target: str = "" # Address of pool to flash from
    flash_fee_bps: int = 0
    profit_usd: Optional[Decimal] = None
    net_profit_usd: Optional[Decimal] = None

# --- RPC & Data Loading ---

def build_rpc_pool(rpc_urls: str) -> List[str]:
    if rpc_urls:
        urls = [normalize_rpc_url(url.strip()) for url in rpc_urls.split(",") if url.strip()]
        return [url for url in urls if url]
    env_url = normalize_rpc_url(os.getenv("ARBITRUM_RPC_URL", "https://arb1.arbitrum.io/rpc"))
    fallbacks = [
        "https://arb1.arbitrum.io/rpc",
        "https://1rpc.io/arb",
        "https://arbitrum.drpc.org",
        "https://arbitrum-one-rpc.publicnode.com",
    ]
    urls = [normalize_rpc_url(u) for u in fallbacks if normalize_rpc_url(u)]
    if env_url and env_url not in urls:
        urls.insert(0, env_url)
    return urls

def fetch_reserves_raw(rpc_url: str, pair: str) -> Tuple[int, int]:
    # 0x0902f1ac = getReserves()
    with httpx.Client(timeout=10) as client:
        resp = client.post(
            rpc_url,
            json={"jsonrpc": "2.0", "id": 1, "method": "eth_call", "params": [{"to": pair, "data": "0x0902f1ac"}, "latest"]},
        )
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload: raise RuntimeError(payload["error"])
    raw = payload["result"][2:]
    return int(raw[0:64], 16), int(raw[64:128], 16)

def to_decimal(raw: int, decimals: int) -> Decimal:
    return Decimal(raw) / (Decimal(10) ** decimals)

def load_pairs(db_path: str, dexes: List[str]) -> List[PairData]:
    conn = sqlite3.connect(db_path)
    dex_placeholders = ",".join("?" for _ in dexes)
    sql = f"""
        SELECT pair_id, token0, token1, token0_symbol, token1_symbol, token0_decimals, token1_decimals, dex
        FROM pairs WHERE dex IN ({dex_placeholders})
    """
    rows = conn.execute(sql, dexes).fetchall()
    pairs = []
    fees = {"uniswapv2": 30, "camelot": 30, "sushiswapv2": 30, "pancakeswapv2": 25}
    for row in rows:
        pairs.append(PairData(
            dex=row[7],
            pair_id=row[0],
            token0=row[1],
            token1=row[2],
            token0_symbol=row[3] or "UNK",
            token1_symbol=row[4] or "UNK",
            token0_decimals=int(row[5] or 18),
            token1_decimals=int(row[6] or 18),
            fee_bps=fees.get(row[7], 30)
        ))
    return pairs

# --- Graph & Path Finding ---

def build_adjacency(pairs: List[PairData]) -> Dict[str, List[PairData]]:
    adj = {}
    for p in pairs:
        adj.setdefault(p.token0.lower(), []).append(p)
        adj.setdefault(p.token1.lower(), []).append(p)
    return adj

def find_cycles(
    start_token: str,
    adj: Dict[str, List[PairData]],
    max_hops: int
) -> List[List[PathStep]]:
    # DFS for cycles
    cycles = []

    # Stack: (current_token, path_so_far, visited_pairs)
    stack = [(start_token.lower(), [], set())]

    while stack:
        curr, path, visited_pairs = stack.pop()

        if len(path) > 0 and curr == start_token.lower():
            cycles.append(path)
            continue

        if len(path) >= max_hops:
            continue

        for pair in adj.get(curr, []):
            if pair.pair_id in visited_pairs:
                continue

            next_token = pair.other_token(curr)

            # Optimization: Don't visit start_token again unless it's the end (which is checked above, but maybe implicitly)
            # Actually, if next_token is start_token, we should push it to check condition next iter?
            # Or just check here.

            # Also pruning: Don't go back to a token we just came from (A->B->A is 2 hop, valid. A->B->A->... invalid loop within loop?)
            # Standard arb is simple cycle. No repeated vertices except start/end.

            seen_tokens = {start_token.lower()} # Start is seen
            for step in path:
                seen_tokens.add(step.token_out.lower()) # token_out of previous steps

            if next_token.lower() in seen_tokens and next_token.lower() != start_token.lower():
                continue # Repeated vertex (not start)

            new_path = path + [PathStep(pair, curr, next_token)]
            new_visited = visited_pairs | {pair.pair_id}
            stack.append((next_token.lower(), new_path, new_visited))

    return cycles

# --- Math & Optimization ---

def get_amount_out(amount_in: Decimal, reserve_in: Decimal, reserve_out: Decimal, fee_bps: int) -> Decimal:
    if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0: return Decimal(0)
    amount_in_with_fee = amount_in * (Decimal(10000) - fee_bps)
    numerator = amount_in_with_fee * reserve_out
    denominator = (reserve_in * Decimal(10000)) + amount_in_with_fee
    return numerator / denominator

def get_path_amount_out(amount_in: Decimal, path: List[PathStep]) -> Decimal:
    current_amount = amount_in
    for step in path:
        r_in, r_out = step.pair.get_reserves(step.token_in)
        current_amount = get_amount_out(current_amount, r_in, r_out, step.pair.fee_bps)
    return current_amount

def optimize_amount_in(path: List[PathStep], max_res_fraction: Decimal = Decimal("0.5")) -> Tuple[Decimal, Decimal]:
    # Ternary search
    # Max input bounded by first pair reserves
    first_step = path[0]
    r_in, _ = first_step.pair.get_reserves(first_step.token_in)
    max_in = r_in * max_res_fraction

    if max_in <= 0: return Decimal(0), Decimal(0)

    def profit_func(amt):
        out = get_path_amount_out(amt, path)
        return out - amt

    lo = Decimal(0)
    hi = max_in
    for _ in range(30):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        p1 = profit_func(m1)
        p2 = profit_func(m2)
        if p1 < p2:
            lo = m1
        else:
            hi = m2

    best_in = (lo + hi) / 2
    best_profit = profit_func(best_in)
    return best_in, best_profit

# --- Execution Simulation ---

def check_flash_sources(
    start_token: str,
    amount: Decimal,
    path_pairs: Set[str],
    all_pairs_with_token: List[PairData],
    rpc_url: str
) -> Tuple[str, str, int]:
    # Returns (source_type, target_address, fee_bps)
    # Types: "aave", "v2_flash", "none"

    best_source = "none"
    best_target = ""
    best_fee = 99999

    # 1. Check Aave
    if start_token.lower() in AAVE_SUPPORTED:
        # Aave fee ~5 bps (0.05%)
        # Actually it's 5 bps on V3
        return "aave", AAVE_V3_POOL_ADDRESS, 5

    # 2. Check V2 Flash Swaps
    # We need a pair that has `start_token`, is NOT in `path_pairs`, and has `reserves > amount`
    # We assume standard 30bps fee (0.3%)

    # We need reserves for these candidates.
    # To save RPC calls, we'll optimistically pick the one with highest reserve?
    # Or just check them.

    # Filter candidates
    candidates = []
    for p in all_pairs_with_token:
        if p.pair_id in path_pairs: continue
        candidates.append(p)

    if not candidates:
        return "none", "", 0

    # Sort candidates by nothing (we don't know reserves yet), maybe random?
    # Or just try the first few.
    for p in candidates[:3]: # check top 3 candidates
        try:
            r0, r1 = fetch_reserves_raw(rpc_url, p.pair_id)
            res = to_decimal(r0 if p.token0.lower() == start_token.lower() else r1, p.token0_decimals if p.token0.lower() == start_token.lower() else p.token1_decimals)
            if res > amount:
                # Valid source
                return "v2_flash", p.pair_id, 30
        except:
            continue

    return "none", "", 0

def construct_execution_steps(arb: ArbitragePath, w3: Web3) -> List[Dict]:
    # Inner swaps
    inner_steps = []
    for step in arb.steps:
        # V2 Swap Step
        # Check if pair uses standard fee?
        extra = b""
        if step.pair.fee_bps != 30:
            extra = encode(['uint256'], [step.pair.fee_bps])

        # AmountIn is 0 for all except first step?
        # Actually, in Monstrosity, if amountIn=0, it uses balance.
        # For the first step inside the flash loan, we have the flash loan funds.
        # So amountIn should be 0 (use balance) OR specific amount.
        # To be safe, first step uses specific amount to ensure we don't use dust?
        # But if we flash borrow X, we have X.
        # Let's use specific amount for first step, 0 for others.

        amt = 0
        if step == arb.steps[0]:
            # Convert decimal to int
            decimals = step.pair.token0_decimals if step.pair.token0.lower() == step.token_in.lower() else step.pair.token1_decimals
            amt = int(arb.optimal_amount_in * (Decimal(10)**decimals))

        s = {
            "action": ACTION_V2_SWAP,
            "target": Web3.to_checksum_address(step.pair.pair_id),
            "tokenIn": Web3.to_checksum_address(step.token_in),
            "tokenOut": Web3.to_checksum_address(step.token_out),
            "amountIn": amt,
            "minAmountOut": 0,
            "extraData": extra
        }
        inner_steps.append(s)

    # Encode inner steps
    step_type = "(uint8,address,address,address,uint256,uint256,bytes)"
    encoded_inner = encode([f"{step_type}[]"], [[
        (s["action"], s["target"], s["tokenIn"], s["tokenOut"], s["amountIn"], s["minAmountOut"], s["extraData"])
        for s in inner_steps
    ]])

    # Outer Flash Step
    raw_amount = int(arb.optimal_amount_in * (Decimal(10) ** (arb.steps[0].pair.token0_decimals if arb.steps[0].pair.token0.lower() == arb.start_token.lower() else arb.steps[0].pair.token1_decimals)))

    action = 0
    if arb.flash_source == "aave":
        action = ACTION_AAVE_FLASH
    elif arb.flash_source == "v2_flash":
        action = ACTION_V2_FLASH_SWAP
    else:
        raise ValueError("Unknown flash source")

    outer_step = {
        "action": action,
        "target": Web3.to_checksum_address(arb.flash_target),
        "tokenIn": Web3.to_checksum_address(arb.start_token),
        "tokenOut": Web3.to_checksum_address(arb.start_token),
        "amountIn": raw_amount,
        "minAmountOut": 0,
        "extraData": encoded_inner
    }

    return [outer_step]

def execute_monstrosity(
    w3: Web3,
    steps: List[Dict],
    private_key: str,
    gas_price_gwei: Decimal
) -> bool:
    account = w3.eth.account.from_key(private_key)
    contract = w3.eth.contract(address=MONSTROSITY_ADDRESS, abi=MONSTROSITY_ABI)

    formatted_steps = []
    for s in steps:
        formatted_steps.append((
            s["action"],
            s["target"],
            s["tokenIn"],
            s["tokenOut"],
            s["amountIn"],
            s["minAmountOut"],
            s["extraData"]
        ))

    try:
        tx_func = contract.functions.execute(formatted_steps, 0)
        gas_estimate = tx_func.estimate_gas({"from": account.address})
        print(f"Simulation success! Gas: {gas_estimate}")
    except ContractLogicError as e:
        print(f"Simulation failed: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Simulation failed: {e}", file=sys.stderr)
        return False

    nonce = w3.eth.get_transaction_count(account.address)
    tx_params = {
        "from": account.address,
        "nonce": nonce,
        "chainId": 42161,
        "gas": int(gas_estimate * 1.2),
        "gasPrice": int(gas_price_gwei * Decimal(1e9))
    }

    try:
        signed_tx = w3.eth.account.sign_transaction(tx_func.build_transaction(tx_params), private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        print(f"Transaction sent: {tx_hash.hex()}")
        return True
    except Exception as e:
        print(f"Execution error: {e}", file=sys.stderr)
        return False

# --- Main Scan Loop ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="skim_pairs.db")
    parser.add_argument("--min-profit-usd", type=Decimal, default=Decimal("1.00"))
    parser.add_argument("--gas-price-gwei", type=Decimal, default=Decimal("0.02"))
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--rpc-urls", default=os.getenv("ARBITRUM_RPC_URLS", ""))
    args = parser.parse_args()

    print("Loading pairs...")
    dexes = ["uniswapv2", "camelot", "sushiswapv2"]
    pairs = load_pairs(args.db_path, dexes)
    adj = build_adjacency(pairs)
    print(f"Loaded {len(pairs)} pairs. Graph size: {len(adj)} tokens.")

    rpc_urls = build_rpc_pool(args.rpc_urls)
    w3 = Web3(Web3.HTTPProvider(rpc_urls[0]))

    iteration = 0
    while True:
        iteration += 1
        print(f"--- Universal Scan Iteration {iteration} ---")

        # We need to scan cycles.
        # Strategy: Iterate over tokens that have enough connections to form cycles.
        sorted_tokens = sorted(adj.keys(), key=lambda t: len(adj[t]), reverse=True)

        # Cache reserves
        reserve_cache = {} # pair_id -> (r0, r1)

        # WETH Price for USD conversion
        # Use WETH-USDC pair
        # Fallback 2000 if fails
        weth_price = Decimal(2500)

        count = 0
        for start_token in sorted_tokens[:100]: # Scan top 100 most connected tokens
            cycles = find_cycles(start_token, adj, args.max_hops)

            for path in cycles:
                # Need reserves for path
                try:
                    for step in path:
                        if step.pair.pair_id not in reserve_cache:
                            r0, r1 = fetch_reserves_raw(rpc_urls[0], step.pair.pair_id)
                            reserve_cache[step.pair.pair_id] = (r0, r1)
                        r0, r1 = reserve_cache[step.pair.pair_id]
                        step.pair.reserve0 = to_decimal(r0, step.pair.token0_decimals)
                        step.pair.reserve1 = to_decimal(r1, step.pair.token1_decimals)
                except Exception:
                    continue

                amt_in, profit = optimize_amount_in(path)
                if profit <= 0: continue

                # We have a profitable swap path. Now find capital.
                path_pair_ids = {s.pair.pair_id for s in path}
                source, target, fee_bps = check_flash_sources(
                    start_token, amt_in, path_pair_ids, adj[start_token], rpc_urls[0]
                )

                if source == "none": continue

                # Calculate Net Profit
                # Gross Profit = Profit - Flash Fee
                flash_fee = amt_in * (Decimal(fee_bps) / Decimal(10000))
                net_token_profit = profit - flash_fee

                if net_token_profit <= 0: continue

                # Convert to USD
                # Need price of start_token
                # Simplified: if start_token is WETH or stable, easy. Else approximate.
                # Just use WETH path if needed.
                # For now, assume WETH 2500.
                # If start_token is WETH
                price = Decimal(0)
                if start_token.lower() == WETH.lower():
                    price = weth_price
                elif start_token.lower() in STABLES:
                    price = Decimal(1)
                else:
                    # skip complex price fetch for speed in this demo
                    price = Decimal(0)

                profit_usd = net_token_profit * price

                # Gas Cost
                # Est 500k gas
                gas_eth = Decimal(500000) * args.gas_price_gwei / Decimal(1e9)
                gas_usd = gas_eth * weth_price

                final_usd = profit_usd - gas_usd

                if final_usd > args.min_profit_usd:
                    print(f"FOUND: {start_token} -> ... -> {start_token} ({len(path)} hops)")
                    print(f"  Source: {source} (Fee: {fee_bps}bps)")
                    print(f"  AmtIn: {amt_in:.4f} | Profit: {net_token_profit:.4f} | Est USD: ${final_usd:.2f}")

                    # Construct Execution
                    arb = ArbitragePath(
                        steps=path,
                        start_token=start_token,
                        optimal_amount_in=amt_in,
                        flash_source=source,
                        flash_target=target
                    )

                    exec_steps = construct_execution_steps(arb, w3)
                    pk = os.getenv("SKIM_PRIVATE_KEY")
                    if pk:
                        execute_monstrosity(w3, exec_steps, pk, args.gas_price_gwei)

        if not args.loop: break
        time.sleep(1)

if __name__ == "__main__":
    main()

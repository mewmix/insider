import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Tuple, Set

import httpx
from dotenv import load_dotenv
from web3 import Web3
from ignore_list import parse_ignore_addresses, is_ignored_address
from policy import parse_allow_addresses, is_allowed_address
from web3.exceptions import ContractLogicError
from eth_abi import encode as eth_abi_encode

from scanner_config import RPC_ENDPOINTS, normalize_rpc_url

load_dotenv()
getcontext().prec = 60

UNISWAP_V2_SUBGRAPH = os.getenv(
    "UNISWAP_V2_SUBGRAPH",
    "https://gateway.thegraph.com/api/subgraphs/id/CStW6CSQbHoXsgKuVCrk3uShGA4JX3CAzzv2x9zaGf8w",
)
CAMELOT_V2_SUBGRAPH = os.getenv(
    "CAMELOT_V2_SUBGRAPH",
    "https://gateway.thegraph.com/api/subgraphs/id/8zagLSufxk5cVhzkzai3tyABwJh53zxn9tmUYJcJxijG",
)
SUSHISWAP_V2_SUBGRAPH = os.getenv(
    "SUSHISWAP_V2_SUBGRAPH",
    "https://gateway.thegraph.com/api/subgraphs/id/8yBXBTMfdhsoE5QCf7KnoPmQb7QAWtRzESfYjiCjGEM9",
)
GRAPH_API_KEY = os.getenv("GRAPH_API_KEY")

WETH = "0x82af49447d8a07e3bd95bd0d56f35241523fbab1"
STABLES = {
    "0xff970a61a04b1ca14834a43f5de4533ebddb5cc8",  # USDC.e
    "0xaf88d065e77c8cc2239327c5edb3a432268e5831",  # USDC
    "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9",  # USDT
    "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1",  # DAI
}

MONSTROSITY_ADDRESS = os.getenv(
    "MONSTROSITY_ADDRESS",
    "0x7e5E849D5a3FBAea7044b4b9e47baBb3d6A60283",
)
AAVE_V3_POOL_ADDRESS = os.getenv(
    "AAVE_POOL_ADDRESS",
    "0x794a61358D6845594F94dc1DB02A252b5b4814aD",
)

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

# Action constants
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
    reserve0: Decimal
    reserve1: Decimal


BALANCE_OF_SIG = "70a08231"
GET_RESERVES_SIG = "0902f1ac"


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
    for name, url in RPC_ENDPOINTS.items():
        if url and not url.startswith("wss://") and "${" not in url:
            if url not in fallbacks:
                fallbacks.append(url)

    urls = [normalize_rpc_url(u) for u in fallbacks if normalize_rpc_url(u)]
    if env_url and env_url not in urls:
        urls.insert(0, env_url)
    return urls


def rpc_call(url: str, method: str, params: List[object]) -> str:
    with httpx.Client(timeout=20) as client:
        resp = client.post(
            url,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        )
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload["result"]


def decode_uint256(hexdata: str) -> int:
    if hexdata.startswith("0x"):
        hexdata = hexdata[2:]
    return int(hexdata or "0", 16)


def fetch_reserves_raw(rpc_url: str, pair: str) -> Tuple[int, int]:
    result = rpc_call(rpc_url, "eth_call", [{"to": pair, "data": "0x" + GET_RESERVES_SIG}, "latest"])
    raw = result[2:]
    if len(raw) < 128:
        raise RuntimeError("Unexpected getReserves response")
    return int(raw[0:64], 16), int(raw[64:128], 16)


def fetch_reserves_with_rotation(
    rpc_urls: List[str], pair: str, start_idx: int
) -> Tuple[int, int]:
    last_exc = None
    for offset in range(len(rpc_urls)):
        rpc_url = rpc_urls[(start_idx + offset) % len(rpc_urls)]
        try:
            return fetch_reserves_raw(rpc_url, pair)
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(last_exc)


def gql_post(url: str, query: str, variables: Dict[str, object]) -> Dict[str, object]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if "gateway.thegraph.com" in url:
        if not GRAPH_API_KEY:
            raise RuntimeError("GRAPH_API_KEY is required for gateway.thegraph.com")
        headers["Authorization"] = f"Bearer {GRAPH_API_KEY}"
    with httpx.Client(timeout=30) as client:
        resp = client.post(url, json={"query": query, "variables": variables}, headers=headers)
    resp.raise_for_status()
    payload = resp.json()
    if "errors" in payload:
        raise RuntimeError(payload["errors"])
    return payload["data"]


def fetch_top_pairs(subgraph: str, order_by: str, first: int) -> List[Tuple[str, str]]:
    query = """
    query TopPairs($first: Int!) {
      pairs(first: $first, orderBy: %s, orderDirection: desc) {
        token0 { id }
        token1 { id }
      }
    }
    """ % order_by
    data = gql_post(subgraph, query, {"first": first})
    pairs = []
    for row in data.get("pairs", []):
        pairs.append((row["token0"]["id"].lower(), row["token1"]["id"].lower()))
    return pairs


def to_decimal(raw: int, decimals: int) -> Decimal:
    return Decimal(raw) / (Decimal(10) ** decimals)


def load_pairs(conn: sqlite3.Connection, dex: str, limit: int) -> List[PairData]:
    sql = """
        SELECT pair_id, token0, token1, token0_symbol, token1_symbol, token0_decimals, token1_decimals
        FROM pairs
        WHERE dex = ?
        ORDER BY pair_id ASC
    """
    params: List[object] = [dex]
    if limit > 0:
        sql += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    pairs = []
    for row in rows:
        pairs.append(
            PairData(
                dex=dex,
                pair_id=row[0],
                token0=row[1],
                token1=row[2],
                token0_symbol=row[3] or "UNKNOWN",
                token1_symbol=row[4] or "UNKNOWN",
                token0_decimals=int(row[5] or 18),
                token1_decimals=int(row[6] or 18),
                reserve0=Decimal(0),
                reserve1=Decimal(0),
            )
        )
    return pairs


def load_pairs_from_watchlist(
    conn: sqlite3.Connection, watchlist_path: str
) -> List[PairData]:
    with open(watchlist_path, "r") as f:
        addresses = json.load(f)
    if not addresses:
        return []

    placeholders = ",".join("?" for _ in addresses)
    sql = f"""
        SELECT pair_id, token0, token1, token0_symbol, token1_symbol, token0_decimals, token1_decimals, dex
        FROM pairs
        WHERE pair_id IN ({placeholders})
    """
    rows = conn.execute(sql, addresses).fetchall()
    pairs = []
    for row in rows:
        pairs.append(
            PairData(
                dex=row[7],
                pair_id=row[0],
                token0=row[1],
                token1=row[2],
                token0_symbol=row[3] or "UNKNOWN",
                token1_symbol=row[4] or "UNKNOWN",
                token0_decimals=int(row[5] or 18),
                token1_decimals=int(row[6] or 18),
                reserve0=Decimal(0),
                reserve1=Decimal(0),
            )
        )
    return pairs


def build_pair_index(pairs: List[PairData]) -> Dict[Tuple[str, str], List[PairData]]:
    index: Dict[Tuple[str, str], List[PairData]] = {}
    for pair in pairs:
        key = tuple(sorted([pair.token0.lower(), pair.token1.lower()]))
        index.setdefault(key, []).append(pair)
    return index


def price_from_reserves(
    token: str,
    token_decimals: int,
    other: str,
    other_decimals: int,
    reserve_token: int,
    reserve_other: int,
) -> Optional[Decimal]:
    if reserve_token <= 0 or reserve_other <= 0:
        return None
    amt_token = to_decimal(reserve_token, token_decimals)
    amt_other = to_decimal(reserve_other, other_decimals)
    if amt_token <= 0:
        return None
    return amt_other / amt_token


def find_best_price_pair(
    token: str,
    other_tokens: List[str],
    pair_index: Dict[Tuple[str, str], List[PairData]],
    rpc_urls: List[str],
    start_idx: int,
) -> Tuple[Optional[Decimal], Optional[str]]:
    best_price = None
    best_other = None
    best_liq = Decimal(0)
    for other in other_tokens:
        key = tuple(sorted([token.lower(), other.lower()]))
        if key not in pair_index:
            continue
        for pair in pair_index[key]:
            try:
                r0, r1 = fetch_reserves_with_rotation(rpc_urls, pair.pair_id, start_idx)
            except Exception:
                continue
            if pair.token0.lower() == token.lower():
                price = price_from_reserves(
                    token,
                    pair.token0_decimals,
                    other,
                    pair.token1_decimals,
                    r0,
                    r1,
                )
                liquidity = to_decimal(r1, pair.token1_decimals)
            else:
                price = price_from_reserves(
                    token,
                    pair.token1_decimals,
                    other,
                    pair.token0_decimals,
                    r1,
                    r0,
                )
                liquidity = to_decimal(r0, pair.token0_decimals)
            if price is None:
                continue
            if liquidity > best_liq:
                best_liq = liquidity
                best_price = price
                best_other = other
    return best_price, best_other


def token_price_usd(
    token: str,
    pair_index: Dict[Tuple[str, str], List[PairData]],
    rpc_urls: List[str],
    start_idx: int,
    weth_price: Optional[Decimal],
) -> Optional[Decimal]:
    token = token.lower()
    if token in STABLES:
        return Decimal(1)
    if token == WETH.lower():
        return weth_price
    stable_list = list(STABLES)
    price, _ = find_best_price_pair(token, stable_list, pair_index, rpc_urls, start_idx)
    if price is not None:
        return price
    if weth_price is None:
        return None
    price_in_weth, _ = find_best_price_pair(token, [WETH], pair_index, rpc_urls, start_idx)
    if price_in_weth is None:
        return None
    return price_in_weth * weth_price


def pair_liquidity_usd(
    pair: PairData,
    pair_index: Dict[Tuple[str, str], List[PairData]],
    rpc_urls: List[str],
    start_idx: int,
    weth_price: Optional[Decimal],
) -> Optional[Decimal]:
    price0 = token_price_usd(pair.token0, pair_index, rpc_urls, start_idx, weth_price)
    price1 = token_price_usd(pair.token1, pair_index, rpc_urls, start_idx, weth_price)
    if price0 is None or price1 is None:
        return None
    return pair.reserve0 * price0 + pair.reserve1 * price1


def choose_best_pool(
    token_a: str,
    token_b: str,
    pair_index: Dict[Tuple[str, str], List[PairData]],
    rpc_urls: List[str],
    start_idx: int,
    reserve_cache: Dict[str, Tuple[int, int]],
) -> Optional[Tuple[PairData, int, int]]:
    key = tuple(sorted([token_a.lower(), token_b.lower()]))
    if key not in pair_index:
        return None
    best = None
    best_liq = Decimal(0)
    for pair in pair_index[key]:
        if pair.pair_id in reserve_cache:
            r0, r1 = reserve_cache[pair.pair_id]
        else:
            r0, r1 = fetch_reserves_with_rotation(rpc_urls, pair.pair_id, start_idx)
            reserve_cache[pair.pair_id] = (r0, r1)
        if pair.token0.lower() == token_b.lower():
            liq = to_decimal(r0, pair.token0_decimals)
        else:
            liq = to_decimal(r1, pair.token1_decimals)
        if liq > best_liq:
            best_liq = liq
            best = (pair, r0, r1)
    return best


def swap_out_simple(
    amount_in: Decimal,
    reserve_in: Decimal,
    reserve_out: Decimal,
    fee_bps: Decimal,
) -> Decimal:
    if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
        return Decimal(0)
    amount_in_with_fee = amount_in * (Decimal(10000) - fee_bps)
    return (amount_in_with_fee * reserve_out) / (reserve_in * Decimal(10000) + amount_in_with_fee)


def settle_profit(
    token_in: str,
    amount_in: Decimal,
    settle_token: str,
    pair_index: Dict[Tuple[str, str], List[PairData]],
    rpc_urls: List[str],
    start_idx: int,
    reserve_cache: Dict[str, Tuple[int, int]],
    fee_bps: Decimal,
) -> Optional[Tuple[Decimal, str]]:
    token_in_lower = token_in.lower()
    settle_lower = settle_token.lower()
    weth_lower = WETH.lower()

    if token_in_lower == settle_lower:
        return amount_in, "direct"

    # Direct path
    direct_out = Decimal(0)
    best_direct = choose_best_pool(
        token_in, settle_token, pair_index, rpc_urls, start_idx, reserve_cache
    )
    if best_direct:
        pair, r0, r1 = best_direct
        if pair.token0.lower() == token_in_lower:
            r_in = to_decimal(r0, pair.token0_decimals)
            r_out = to_decimal(r1, pair.token1_decimals)
        else:
            r_in = to_decimal(r1, pair.token1_decimals)
            r_out = to_decimal(r0, pair.token0_decimals)
        direct_out = swap_out_simple(amount_in, r_in, r_out, fee_bps)

    # Multi-hop via WETH
    multihop_out = Decimal(0)
    if token_in_lower != weth_lower and settle_lower != weth_lower:
        # Step 1: token_in -> WETH
        step1 = choose_best_pool(
            token_in, WETH, pair_index, rpc_urls, start_idx, reserve_cache
        )
        if step1:
            pair1, r0_1, r1_1 = step1
            if pair1.token0.lower() == token_in_lower:
                r_in1 = to_decimal(r0_1, pair1.token0_decimals)
                r_out1 = to_decimal(r1_1, pair1.token1_decimals)
            else:
                r_in1 = to_decimal(r1_1, pair1.token1_decimals)
                r_out1 = to_decimal(r0_1, pair1.token0_decimals)
            weth_amt = swap_out_simple(amount_in, r_in1, r_out1, fee_bps)

            # Step 2: WETH -> settle_token
            step2 = choose_best_pool(
                WETH, settle_token, pair_index, rpc_urls, start_idx, reserve_cache
            )
            if step2:
                pair2, r0_2, r1_2 = step2
                if pair2.token0.lower() == weth_lower:
                    r_in2 = to_decimal(r0_2, pair2.token0_decimals)
                    r_out2 = to_decimal(r1_2, pair2.token1_decimals)
                else:
                    r_in2 = to_decimal(r1_2, pair2.token1_decimals)
                    r_out2 = to_decimal(r0_2, pair2.token0_decimals)
                multihop_out = swap_out_simple(weth_amt, r_in2, r_out2, fee_bps)

    if direct_out <= 0 and multihop_out <= 0:
        return None

    if multihop_out > direct_out:
        return multihop_out, "multihop"
    return direct_out, "direct"


def swap_out(amount_in: Decimal, reserve_in: Decimal, reserve_out: Decimal, fee: Decimal) -> Decimal:
    if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
        return Decimal(0)
    amount_in_with_fee = amount_in * (Decimal(1) - fee)
    return (amount_in_with_fee * reserve_out) / (reserve_in + amount_in_with_fee)


def ternary_search(
    reserve_in: Decimal,
    reserve_out: Decimal,
    reserve_in_2: Decimal,
    reserve_out_2: Decimal,
    fee_in: Decimal,
    fee_out: Decimal,
    max_in: Decimal,
) -> Tuple[Decimal, Decimal]:
    lo = Decimal(0)
    hi = max_in
    best_in = Decimal(0)
    best_profit = Decimal(0)
    for _ in range(50):
        m1 = lo + (hi - lo) / Decimal(3)
        m2 = hi - (hi - lo) / Decimal(3)
        p1 = profit_for_amount(
            m1, reserve_in, reserve_out, reserve_in_2, reserve_out_2, fee_in, fee_out
        )
        p2 = profit_for_amount(
            m2, reserve_in, reserve_out, reserve_in_2, reserve_out_2, fee_in, fee_out
        )
        if p1 > p2:
            hi = m2
        else:
            lo = m1
        if p1 > best_profit:
            best_profit = p1
            best_in = m1
        if p2 > best_profit:
            best_profit = p2
            best_in = m2
    return best_in, best_profit


def profit_for_amount(
    amount_in: Decimal,
    reserve_in: Decimal,
    reserve_out: Decimal,
    reserve_in_2: Decimal,
    reserve_out_2: Decimal,
    fee_in: Decimal,
    fee_out: Decimal,
) -> Decimal:
    out_1 = swap_out(amount_in, reserve_in, reserve_out, fee_in)
    out_2 = swap_out(out_1, reserve_in_2, reserve_out_2, fee_out)
    return out_2 - amount_in


def ternary_search_generic(
    max_in: Decimal,
    profit_fn,
) -> Tuple[Decimal, Decimal]:
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


def swap_out_pair(
    amount_in: Decimal,
    token_in: str,
    pair: PairData,
    fee: Decimal,
) -> Decimal:
    if pair.token0.lower() == token_in.lower():
        return swap_out(amount_in, pair.reserve0, pair.reserve1, fee)
    return swap_out(amount_in, pair.reserve1, pair.reserve0, fee)


def triangle_out(
    amount_in: Decimal,
    start_token: str,
    token_b: str,
    token_c: str,
    pair_ab: PairData,
    pair_bc: PairData,
    pair_ca: PairData,
    fee_by_dex: Dict[str, Decimal],
) -> Decimal:
    fee_ab = fee_by_dex.get(pair_ab.dex, Decimal("0.003"))
    fee_bc = fee_by_dex.get(pair_bc.dex, Decimal("0.003"))
    fee_ca = fee_by_dex.get(pair_ca.dex, Decimal("0.003"))
    amt = swap_out_pair(amount_in, start_token, pair_ab, fee_ab)
    amt = swap_out_pair(amt, token_b, pair_bc, fee_bc)
    return swap_out_pair(amt, token_c, pair_ca, fee_ca)


def best_triangle_arb(
    start_token: str,
    token_b: str,
    token_c: str,
    pair_ab: PairData,
    pair_bc: PairData,
    pair_ca: PairData,
    fee_by_dex: Dict[str, Decimal],
    max_trade_frac: Decimal,
) -> Tuple[Decimal, Decimal]:
    if pair_ab.token0.lower() == start_token.lower():
        max_in = pair_ab.reserve0 * max_trade_frac
    else:
        max_in = pair_ab.reserve1 * max_trade_frac
    if max_in <= 0:
        return Decimal(0), Decimal(0)

    def profit_fn(amount_in: Decimal) -> Decimal:
        out_amt = triangle_out(
            amount_in, start_token, token_b, token_c, pair_ab, pair_bc, pair_ca, fee_by_dex
        )
        return out_amt - amount_in

    return ternary_search_generic(max_in, profit_fn)


def best_arb_for_token0(
    pool_a: PairData,
    pool_b: PairData,
    fee_a: Decimal,
    fee_b: Decimal,
    max_trade_frac: Decimal,
) -> Tuple[Decimal, Decimal, str]:
    max_in = pool_a.reserve0 * max_trade_frac
    if max_in <= 0:
        return Decimal(0), Decimal(0), ""
    amt_in, profit = ternary_search(
        pool_a.reserve0,
        pool_a.reserve1,
        pool_b.reserve1,
        pool_b.reserve0,
        fee_a,
        fee_b,
        max_in,
    )
    return amt_in, profit, "token0"


def best_arb_for_token1(
    pool_a: PairData,
    pool_b: PairData,
    fee_a: Decimal,
    fee_b: Decimal,
    max_trade_frac: Decimal,
) -> Tuple[Decimal, Decimal, str]:
    max_in = pool_a.reserve1 * max_trade_frac
    if max_in <= 0:
        return Decimal(0), Decimal(0), ""
    amt_in, profit = ternary_search(
        pool_a.reserve1,
        pool_a.reserve0,
        pool_b.reserve0,
        pool_b.reserve1,
        fee_a,
        fee_b,
        max_in,
    )
    return amt_in, profit, "token1"


def execute_monstrosity(
    w3: Web3,
    steps: List[Dict],
    contract_address: str,
    private_key: str,
    gas_price_gwei: Decimal,
    min_profit_weth: int,
    dry_run: bool = False,
) -> bool:
    account = w3.eth.account.from_key(private_key)
    contract = w3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=MONSTROSITY_ABI)

    print(f"Executing with {len(steps)} top-level steps...")

    # Format steps for contract
    formatted_steps = []
    for s in steps:
        formatted_steps.append((
            s["action"],
            Web3.to_checksum_address(s["target"]),
            Web3.to_checksum_address(s["tokenIn"]),
            Web3.to_checksum_address(s.get("tokenOut", "0x0000000000000000000000000000000000000000")),
            s["amountIn"],
            s.get("minAmountOut", 0),
            s.get("extraData", b"")
        ))

    try:
        tx_func = contract.functions.execute(formatted_steps, min_profit_weth)
        gas_estimate = tx_func.estimate_gas({"from": account.address})
        print(f"Simulation success! Gas: {gas_estimate}")
    except ContractLogicError as e:
        print(f"Simulation failed (revert): {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Simulation failed: {e}", file=sys.stderr)
        return False

    if dry_run:
        return True

    print("Submitting transaction...")
    nonce = w3.eth.get_transaction_count(account.address)

    # Simple legacy or EIP-1559 based on inputs.
    # We use user specified gas price or dynamic?
    # User said "Gwei is 0.02" fixed? "Gwei is 0.02" usually implies max fee.
    # But for Arb, we often use dynamic.
    # Let's use user preference if provided strict.

    tx_params = {
        "from": account.address,
        "nonce": nonce,
        "chainId": 42161, # Arbitrum One
        "gas": int(gas_estimate * 1.2),
        "gasPrice": int(gas_price_gwei * Decimal(1e9))
    }

    try:
        signed_tx = w3.eth.account.sign_transaction(tx_func.build_transaction(tx_params), private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        print(f"Transaction sent: {tx_hash.hex()}")
        sys.stdout.flush()
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if receipt.status == 1:
            print("Transaction confirmed!")
            return True
        else:
            print("Transaction reverted on chain.")
            return False
    except Exception as e:
        print(f"Execution error: {e}", file=sys.stderr)
        return False


def execute_trade(
    pair_borrow: str,
    pair_swap: str,
    token_borrow: str,
    amount_borrow: int,
    fee_borrow_bps: int,
    fee_swap_bps: int,
    min_profit: int,
    dry_run: bool,
    rpc_url: str,
    private_key: str,
    gas_price_gwei: Decimal,
    monstrosity_address: str,
    aave_pool_address: str,
) -> bool:
    if not private_key:
        print("skipping execution: no private key", file=sys.stderr)
        return False

    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        return False

    # Logic: Flash Loan (Aave) -> Swap 1 -> Swap 2 -> Repay
    # Or 2-Hop using Aave:
    # 1. Flash Aave (token_borrow)
    # 2. Swap token_borrow -> token_other (on pair_swap - wait, naming convention?)

    # Original logic: Borrow from Pair B, Swap on Pair A.
    # Here: Borrow from Aave.
    # Then Swap on Pair A (A->B), then Swap on Pair B (B->A).
    # Wait, the pairs passed are:
    # pair_borrow: The one we borrowed from in old logic (but here we treat as just one of the pairs)
    # pair_swap: The other pair.
    # If logic was: Borrow from pair_borrow, swap on pair_swap.
    # Route: pair_swap (A->B), pair_borrow (B->A).
    # Correct?
    # In `flash_swap_scanner.py`:
    # candidates.append((f"{dex_a}->{dex_b}", ... pair_a, pair_b ...))
    # execute(pair_borrow=pair_b, pair_swap=pair_a ...)
    # So pair_swap is the FIRST hop (A->B). pair_borrow is the SECOND hop (B->A).

    pair_1 = pair_swap
    pair_2 = pair_borrow

    # We need to know the tokens.
    # token_borrow is the input token.

    # Step 1: Swap on Pair 1 (Input -> Output)
    step1 = {
        "action": ACTION_V2_SWAP,
        "target": pair_1,
        "tokenIn": token_borrow,
        "tokenOut": "0x0000000000000000000000000000000000000000", # ignored by V2 swap logic usually, but good to have
        "amountIn": amount_borrow,
        "minAmountOut": 0,
        "extraData": b"" # Fee? Default 30bps. If we need custom, encode it.
        # But we don't have fee passed easily here other than fee_swap_bps.
        # We should encode fee if it's not 30bps.
        # Encode fee_swap_bps.
    }
    if fee_swap_bps != 30:
        step1["extraData"] = eth_abi_encode(["uint256"], [fee_swap_bps])

    # Step 2: Swap on Pair 2 (Output -> Input)
    # We don't know exact Output amount from Step 1, so we use Balance.
    # TokenIn for Step 2 is the OTHER token of Pair 1.
    # We need to resolve it.
    # But Step struct requires tokenIn.
    # We can infer it or we can look it up.
    # For now, let's assume we can get it from pair data if we had it, but we only have address here.
    # Actually, we can just assume it works if we set amountIn=0 (balance).
    # But Monstrosity needs `tokenIn` to call transfer.

    # We need to fetch Pair 1 token0/1 to know which one is NOT token_borrow.
    pair1_contract = w3.eth.contract(address=Web3.to_checksum_address(pair_1), abi=[
        {"constant":True,"inputs":[],"name":"token0","outputs":[{"name":"","type":"address"}],"type":"function"},
        {"constant":True,"inputs":[],"name":"token1","outputs":[{"name":"","type":"address"}],"type":"function"}
    ])
    p1_t0 = pair1_contract.functions.token0().call()
    p1_t1 = pair1_contract.functions.token1().call()

    token_intermediate = p1_t1 if p1_t0.lower() == token_borrow.lower() else p1_t0

    step2 = {
        "action": ACTION_V2_SWAP,
        "target": pair_2,
        "tokenIn": token_intermediate,
        "tokenOut": token_borrow,
        "amountIn": 0, # Use balance
        "minAmountOut": 0,
        "extraData": b""
    }
    if fee_borrow_bps != 30:
        step2["extraData"] = eth_abi_encode(["uint256"], [fee_borrow_bps])

    # Encoded Nested Steps
    nested_steps = [step1, step2]
    # Encode using eth_abi
    # Struct type: (uint8,address,address,address,uint256,uint256,bytes)
    step_type = "(uint8,address,address,address,uint256,uint256,bytes)"
    encoded_nested = eth_abi_encode([f"{step_type}[]"], [ [
        (s["action"], s["target"], s["tokenIn"], s["tokenOut"], s["amountIn"], s["minAmountOut"], s["extraData"])
        for s in nested_steps
    ] ])

    # Top Level: Aave Flash
    flash_step = {
        "action": ACTION_AAVE_FLASH,
        "target": aave_pool_address,
        "tokenIn": token_borrow,
        "tokenOut": token_borrow, # Same for flash
        "amountIn": amount_borrow,
        "minAmountOut": 0,
        "extraData": encoded_nested
    }

    return execute_monstrosity(
        w3, [flash_step], monstrosity_address, private_key, gas_price_gwei, 0, dry_run
    )


def execute_triangular_trade(
    start_token: str,
    token_b: str,
    token_c: str,
    pair_ab: PairData,
    pair_bc: PairData,
    pair_ca: PairData,
    hop_types: Tuple[str, str, str],
    amount_in: int,
    rpc_url: str,
    private_key: str,
    gas_price_gwei: Decimal,
    dry_run: bool,
    monstrosity_address: str,
    aave_pool_address: str,
    fee_by_dex: Dict[str, Decimal],
    safety_bps: Decimal,
    min_profit_weth: int,
) -> bool:
    if not private_key: return False
    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
    if not w3.is_connected(): return False

    # Steps:
    # 1. Swap start -> b (on pair_ab)
    # 2. Swap b -> c (on pair_bc)
    # 3. Swap c -> start (on pair_ca)

    fee_ab = fee_by_dex.get(pair_ab.dex, Decimal("0.003"))
    fee_bc = fee_by_dex.get(pair_bc.dex, Decimal("0.003"))
    fee_ca = fee_by_dex.get(pair_ca.dex, Decimal("0.003"))

    r0_ab, r1_ab = fetch_reserves_raw(rpc_url, pair_ab.pair_id)
    r0_bc, r1_bc = fetch_reserves_raw(rpc_url, pair_bc.pair_id)
    r0_ca, r1_ca = fetch_reserves_raw(rpc_url, pair_ca.pair_id)

    pair_ab.reserve0 = to_decimal(r0_ab, pair_ab.token0_decimals)
    pair_ab.reserve1 = to_decimal(r1_ab, pair_ab.token1_decimals)
    pair_bc.reserve0 = to_decimal(r0_bc, pair_bc.token0_decimals)
    pair_bc.reserve1 = to_decimal(r1_bc, pair_bc.token1_decimals)
    pair_ca.reserve0 = to_decimal(r0_ca, pair_ca.token0_decimals)
    pair_ca.reserve1 = to_decimal(r1_ca, pair_ca.token1_decimals)

    in_decimals = pair_ab.token0_decimals if pair_ab.token0.lower() == start_token.lower() else pair_ab.token1_decimals
    amount_in_dec = to_decimal(amount_in, in_decimals)
    out1 = swap_out_pair(amount_in_dec, start_token, pair_ab, fee_ab) if hop_types[0] != "v3" else Decimal(0)
    out2 = swap_out_pair(out1, token_b, pair_bc, fee_bc) if hop_types[1] != "v3" else Decimal(0)
    out3 = swap_out_pair(out2, token_c, pair_ca, fee_ca) if hop_types[2] != "v3" else Decimal(0)

    safety_mult = (Decimal(10000) - safety_bps) / Decimal(10000)
    min_out1 = out1 * safety_mult if hop_types[0] != "v3" else Decimal(0)
    min_out2 = out2 * safety_mult if hop_types[1] != "v3" else Decimal(0)
    min_out3 = out3 * safety_mult if hop_types[2] != "v3" else Decimal(0)

    out1_decimals = pair_ab.token1_decimals if pair_ab.token0.lower() == start_token.lower() else pair_ab.token0_decimals
    out2_decimals = pair_bc.token1_decimals if pair_bc.token0.lower() == token_b.lower() else pair_bc.token0_decimals
    out3_decimals = pair_ca.token1_decimals if pair_ca.token0.lower() == token_c.lower() else pair_ca.token0_decimals

    min_out1_raw = int(min_out1 * (Decimal(10) ** out1_decimals))
    min_out2_raw = int(min_out2 * (Decimal(10) ** out2_decimals))
    min_out3_raw = int(min_out3 * (Decimal(10) ** out3_decimals))

    fee_ab_bps = int(fee_ab * Decimal(10000))
    fee_bc_bps = int(fee_bc * Decimal(10000))
    fee_ca_bps = int(fee_ca * Decimal(10000))

    step1 = {
        "action": ACTION_V3_SWAP if hop_types[0] == "v3" else ACTION_V2_SWAP,
        "target": pair_ab.pair_id,
        "tokenIn": start_token,
        "tokenOut": token_b,
        "amountIn": amount_in,
        "minAmountOut": max(0, min_out1_raw),
        "extraData": (eth_abi_encode(["uint256"], [fee_ab_bps]) if fee_ab_bps != 30 else b"") if hop_types[0] != "v3" else b""
    }
    step2 = {
        "action": ACTION_V3_SWAP if hop_types[1] == "v3" else ACTION_V2_SWAP,
        "target": pair_bc.pair_id,
        "tokenIn": token_b,
        "tokenOut": token_c,
        "amountIn": 0,
        "minAmountOut": max(0, min_out2_raw),
        "extraData": (eth_abi_encode(["uint256"], [fee_bc_bps]) if fee_bc_bps != 30 else b"") if hop_types[1] != "v3" else b""
    }
    step3 = {
        "action": ACTION_V3_SWAP if hop_types[2] == "v3" else ACTION_V2_SWAP,
        "target": pair_ca.pair_id,
        "tokenIn": token_c,
        "tokenOut": start_token,
        "amountIn": 0,
        "minAmountOut": max(0, min_out3_raw),
        "extraData": (eth_abi_encode(["uint256"], [fee_ca_bps]) if fee_ca_bps != 30 else b"") if hop_types[2] != "v3" else b""
    }

    nested_steps = [step1, step2, step3]
    step_type = "(uint8,address,address,address,uint256,uint256,bytes)"
    encoded_nested = eth_abi_encode([f"{step_type}[]"], [ [
        (s["action"], s["target"], s["tokenIn"], s["tokenOut"], s["amountIn"], s["minAmountOut"], s["extraData"])
        for s in nested_steps
    ] ])

    flash_step = {
        "action": ACTION_AAVE_FLASH,
        "target": aave_pool_address,
        "tokenIn": start_token,
        "tokenOut": start_token,
        "amountIn": amount_in,
        "minAmountOut": 0,
        "extraData": encoded_nested
    }

    return execute_monstrosity(
        w3,
        [flash_step],
        monstrosity_address,
        private_key,
        gas_price_gwei,
        min_profit_weth,
        dry_run,
    )


def execute_path_trade(
    path_tokens: List[str],
    path_pairs: List[PairData],
    amount_in: int,
    rpc_url: str,
    private_key: str,
    gas_price_gwei: Decimal,
    dry_run: bool,
    monstrosity_address: str,
    aave_pool_address: str,
    fee_by_dex: Dict[str, Decimal],
    safety_bps: Decimal,
    min_profit_weth: int,
) -> bool:
    if not private_key:
        return False
    if len(path_tokens) < 2 or len(path_pairs) != len(path_tokens) - 1:
        return False

    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        return False

    # Fetch reserves for minAmountOut calculations.
    for pair in path_pairs:
        r0, r1 = fetch_reserves_raw(rpc_url, pair.pair_id)
        pair.reserve0 = to_decimal(r0, pair.token0_decimals)
        pair.reserve1 = to_decimal(r1, pair.token1_decimals)

    start_token = path_tokens[0]
    in_decimals = (
        path_pairs[0].token0_decimals
        if path_pairs[0].token0.lower() == start_token.lower()
        else path_pairs[0].token1_decimals
    )
    amount_in_dec = to_decimal(amount_in, in_decimals)

    # Build minAmountOut per hop with safety haircut.
    safety_mult = (Decimal(10000) - safety_bps) / Decimal(10000)
    min_outs_raw: List[int] = []
    cur_amount = amount_in_dec
    for idx, pair in enumerate(path_pairs):
        fee = fee_by_dex.get(pair.dex, Decimal("0.003"))
        token_in = path_tokens[idx]
        token_out = path_tokens[idx + 1]
        cur_amount = swap_out_pair(cur_amount, token_in, pair, fee)
        min_out = cur_amount * safety_mult
        out_decimals = (
            pair.token1_decimals
            if pair.token0.lower() == token_in.lower()
            else pair.token0_decimals
        )
        min_outs_raw.append(int(min_out * (Decimal(10) ** out_decimals)))

    steps: List[Dict] = []
    for idx, pair in enumerate(path_pairs):
        is_v3 = "v3" in pair.dex
        action = ACTION_V3_SWAP if is_v3 else ACTION_V2_SWAP

        # Calculate fee extra data for V2 only
        fee_extra = b""
        if not is_v3:
            fee = fee_by_dex.get(pair.dex, Decimal("0.003"))
            fee_bps = int(fee * Decimal(10000))
            if fee_bps != 30:
                fee_extra = eth_abi_encode(["uint256"], [fee_bps])

        steps.append(
            {
                "action": action,
                "target": pair.pair_id,
                "tokenIn": path_tokens[idx],
                "tokenOut": path_tokens[idx + 1],
                "amountIn": amount_in if idx == 0 else 0,
                "minAmountOut": max(0, min_outs_raw[idx]),
                "extraData": fee_extra,
            }
        )

    step_type = "(uint8,address,address,address,uint256,uint256,bytes)"
    encoded_nested = eth_abi_encode(
        [f"{step_type}[]"],
        [[
            (
                s["action"],
                s["target"],
                s["tokenIn"],
                s["tokenOut"],
                s["amountIn"],
                s["minAmountOut"],
                s["extraData"],
            )
            for s in steps
        ]],
    )

    flash_step = {
        "action": ACTION_AAVE_FLASH,
        "target": aave_pool_address,
        "tokenIn": start_token,
        "tokenOut": start_token,
        "amountIn": amount_in,
        "minAmountOut": 0,
        "extraData": encoded_nested,
    }

    return execute_monstrosity(
        w3,
        [flash_step],
        monstrosity_address,
        private_key,
        gas_price_gwei,
        min_profit_weth,
        dry_run,
    )


def deep_scan_cycles(
    adj: Dict[str, List[Tuple[str, PairData]]],
    start_token: str,
    min_hops: int,
    max_hops: int,
    max_paths: int = 50
) -> List[List[Tuple[str, PairData]]]:
    results = []
    # stack: (current_token, path_so_far, visited_set)
    # path_so_far: List[Tuple[destination_token, pair_used]]
    stack = [(start_token, [], {start_token})]

    while stack:
        curr, path, visited = stack.pop()

        if len(path) >= max_hops:
            continue

        if curr not in adj:
            continue

        for neighbor, pair in adj[curr]:
            if neighbor == start_token:
                if len(path) + 1 >= min_hops:
                    results.append(path + [(neighbor, pair)])
                    if len(results) >= max_paths:
                        return results
                continue

            if neighbor in visited:
                continue

            new_visited = visited | {neighbor}
            new_path = path + [(neighbor, pair)]
            stack.append((neighbor, new_path, new_visited))

    return results


def complex_scan(
    pairs: List[PairData],
    pair_index: Dict[Tuple[str, str], List[PairData]],
    rpc_urls: List[str],
    start_idx: int,
    reserve_cache: Dict[str, Tuple[int, int]],
    weth_price: Optional[Decimal],
    gas_units: int,
    gas_price_gwei: Decimal,
    min_net_profit_usd: Decimal,
    fee_by_dex: Dict[str, Decimal],
    max_trade_frac: Decimal,
    auto_execute: bool,
    safety_bps: Decimal,
    monstrosity_address: str,
    aave_pool_address: str,
    ignore_addresses: Set[str],
    dump_path: str,
    allow_addresses: Set[str],
    allow_any: bool,
    min_pair_liquidity_usd: Decimal,
    simulate_all: bool,
    allow_v3: bool,
    v3_amount_in: Decimal,
    min_hops: int,
    max_hops: int,
) -> None:
    adj: Dict[str, List[Tuple[str, PairData]]] = {}
    for p in pairs:
        t0 = p.token0.lower()
        t1 = p.token1.lower()
        if allow_addresses:
            if not (is_allowed_address(t0, allow_addresses, allow_any=False) or is_allowed_address(t1, allow_addresses, allow_any=False)):
                continue
        adj.setdefault(t0, []).append((t1, p))
        adj.setdefault(t1, []).append((t0, p))

    sorted_tokens = sorted(adj.keys(), key=lambda k: len(adj[k]), reverse=True)
    print(f"Scanning complex arb opportunities (hops {min_hops}-{max_hops})...")

    count = 0
    # Limit start tokens to top 200 for performance unless restricted by allow_addresses
    scan_list = sorted_tokens if allow_addresses else sorted_tokens[:200]

    for start_token in scan_list:
        if start_token in ignore_addresses:
            continue
        if allow_addresses and not is_allowed_address(start_token, allow_addresses, allow_any=False):
            continue

        cycles = deep_scan_cycles(adj, start_token, min_hops, max_hops)

        for cycle in cycles:
            # cycle: List[(next_token, pair)]
            path_tokens = [start_token] + [x[0] for x in cycle]
            path_pairs = [x[1] for x in cycle]

            if any(t in ignore_addresses for t in path_tokens): continue
            if any(p.pair_id in ignore_addresses for p in path_pairs): continue

            hop_types = tuple("v3" if "v3" in p.dex else "v2" for p in path_pairs)
            has_v3 = any(h == "v3" for h in hop_types)
            if has_v3 and not allow_v3:
                continue

            # Update reserves
            for p in path_pairs:
                if "v3" in p.dex: continue
                if p.pair_id not in reserve_cache:
                    try:
                        r0, r1 = fetch_reserves_with_rotation(rpc_urls, p.pair_id, start_idx)
                        reserve_cache[p.pair_id] = (r0, r1)
                        p.reserve0 = to_decimal(r0, p.token0_decimals)
                        p.reserve1 = to_decimal(r1, p.token1_decimals)
                    except Exception:
                        pass

            estimate_ok = not has_v3
            profit_safe = Decimal(0)
            net_profit_usd = None
            amt_in = Decimal(0)

            if estimate_ok:
                # We need a generic profit estimator for N hops.
                # Use ternary search on the path function?
                # Defining profit_fn closure

                # Check liquidity first
                max_in_start = Decimal(0)
                p0 = path_pairs[0]
                if p0.token0.lower() == start_token.lower():
                    max_in_start = p0.reserve0 * max_trade_frac
                else:
                    max_in_start = p0.reserve1 * max_trade_frac

                if max_in_start <= 0: continue

                def profit_fn_path(amount_in_val: Decimal) -> Decimal:
                    cur = amount_in_val
                    for i, p in enumerate(path_pairs):
                        t_in = path_tokens[i]
                        fee = fee_by_dex.get(p.dex, Decimal("0.003"))
                        cur = swap_out_pair(cur, t_in, p, fee)
                    return cur - amount_in_val

                amt_in, profit = ternary_search_generic(max_in_start, profit_fn_path)

                if profit <= 0: continue

                # Safety haircut
                safety_mult = (Decimal(10000) - safety_bps) / Decimal(10000)
                amt_out = amt_in + profit
                amt_out_safe = amt_out * safety_mult
                profit_safe = amt_out_safe - amt_in

                if profit_safe <= 0: continue

                price_usd = token_price_usd(start_token, pair_index, rpc_urls, start_idx, weth_price)
                profit_usd = None if price_usd is None else profit_safe * price_usd

                if profit_usd is not None and weth_price is not None:
                    # Gas estimate: ~150k + 100k per hop?
                    est_gas = 150000 + (100000 * len(path_pairs))
                    gas_cost_eth = (Decimal(est_gas) * gas_price_gwei) / Decimal(1e9)
                    gas_cost_usd = gas_cost_eth * weth_price
                    net_profit_usd = profit_usd - gas_cost_usd

                if min_net_profit_usd > 0 and (net_profit_usd is None or net_profit_usd < min_net_profit_usd):
                    continue

                if min_pair_liquidity_usd > 0:
                    low_liq = False
                    for p in path_pairs:
                        liq = pair_liquidity_usd(p, pair_index, rpc_urls, start_idx, weth_price)
                        if liq is None or liq < min_pair_liquidity_usd:
                            low_liq = True
                            break
                    if low_liq: continue

            else:
                if start_token.lower() != WETH.lower(): continue
                amt_in = v3_amount_in

            min_profit_weth = Decimal(0)
            if weth_price is not None and min_net_profit_usd > 0 and start_token.lower() == WETH.lower():
                min_profit_weth = min_net_profit_usd / weth_price
            min_profit_weth_raw = int(min_profit_weth * Decimal(10) ** 18) if min_profit_weth > 0 else 0

            decimals = path_pairs[0].token0_decimals if path_pairs[0].token0.lower() == start_token.lower() else path_pairs[0].token1_decimals
            raw_amount_in = int(amt_in * (Decimal(10) ** decimals))

            sim_ok = False
            if simulate_all or auto_execute:
                if min_net_profit_usd > 0 and start_token.lower() != WETH.lower():
                    continue

                sim_ok = execute_path_trade(
                    path_tokens, path_pairs,
                    raw_amount_in,
                    rpc_urls[0],
                    os.getenv("SKIM_PRIVATE_KEY", ""),
                    gas_price_gwei,
                    dry_run=True,
                    monstrosity_address=monstrosity_address,
                    aave_pool_address=aave_pool_address,
                    fee_by_dex=fee_by_dex,
                    safety_bps=safety_bps,
                    min_profit_weth=min_profit_weth_raw
                )
                if simulate_all:
                    status = "ok" if sim_ok else "fail"
                    dex_path = "->".join(p.dex for p in path_pairs)
                    print(f"SIM {status}: {'->'.join(path_tokens)} ({dex_path}) | in={amt_in:.6f}")

            if estimate_ok:
                net_note = f"net=${net_profit_usd:.2f}" if net_profit_usd is not None else "net=unknown"
                dex_path = "->".join(p.dex for p in path_pairs)
                print(
                    f"FOUND COMPLEX ARB: {'->'.join(path_tokens)} ({dex_path}) | "
                    f"in={amt_in:.6f} profit={profit_safe:.6f} | {net_note}"
                )
            elif sim_ok:
                dex_path = "->".join(p.dex for p in path_pairs)
                print(f"FOUND COMPLEX ARB (sim-only): {'->'.join(path_tokens)} ({dex_path}) | in={amt_in:.6f}")

            if sim_ok and auto_execute:
                if not allow_any and not allow_addresses: continue
                execute_path_trade(
                    path_tokens, path_pairs,
                    raw_amount_in,
                    rpc_urls[0],
                    os.getenv("SKIM_PRIVATE_KEY", ""),
                    gas_price_gwei,
                    dry_run=False,
                    monstrosity_address=monstrosity_address,
                    aave_pool_address=aave_pool_address,
                    fee_by_dex=fee_by_dex,
                    safety_bps=safety_bps,
                    min_profit_weth=min_profit_weth_raw
                )

            count += 1
            if count > 10: break # per iteration limit
        if count > 10: break


def triangular_scan(
    pairs: List[PairData],
    pair_index: Dict[Tuple[str, str], List[PairData]],
    rpc_urls: List[str],
    start_idx: int,
    reserve_cache: Dict[str, Tuple[int, int]],
    weth_price: Optional[Decimal],
    gas_units: int,
    gas_price_gwei: Decimal,
    min_net_profit_usd: Decimal,
    fee_by_dex: Dict[str, Decimal],
    max_trade_frac: Decimal,
    auto_execute: bool,
    safety_bps: Decimal,
    monstrosity_address: str,
    aave_pool_address: str,
    ignore_addresses: Set[str],
    dump_path: str,
    allow_addresses: Set[str],
    allow_any: bool,
    min_pair_liquidity_usd: Decimal,
    simulate_all: bool,
    allow_v3: bool,
    v3_amount_in: Decimal,
) -> None:
    adj: Dict[str, List[Tuple[str, PairData]]] = {}
    for p in pairs:
        t0 = p.token0.lower()
        t1 = p.token1.lower()
        if allow_addresses:
            if not (is_allowed_address(t0, allow_addresses, allow_any=False) or is_allowed_address(t1, allow_addresses, allow_any=False)):
                continue
        adj.setdefault(t0, []).append((t1, p))
        adj.setdefault(t1, []).append((t0, p))

    sorted_tokens = sorted(adj.keys(), key=lambda k: len(adj[k]), reverse=True)
    print(f"Scanning triangular arb opportunities (Graph size: {len(adj)} tokens)...")

    count = 0
    # Wide scan: Check top 1000 tokens (or all if feasible)
    for start_token in sorted_tokens[:1000]:
        if start_token in ignore_addresses:
            continue
        if allow_addresses and not is_allowed_address(start_token, allow_addresses, allow_any=False):
            continue
        for (b, pair_ab) in adj[start_token]:
            if b == start_token: continue
            if b in ignore_addresses:
                continue
            if allow_addresses and not is_allowed_address(b, allow_addresses, allow_any=False):
                continue
            for (c, pair_bc) in adj[b]:
                if c == start_token or c == b: continue
                if c in ignore_addresses:
                    continue
                if allow_addresses and not is_allowed_address(c, allow_addresses, allow_any=False):
                    continue
                for (d, pair_ca) in adj[c]:
                    if d == start_token:
                        if any(tok in ignore_addresses for tok in (start_token, b, c)):
                            continue
                        if any(
                            is_ignored_address(p.pair_id, ignore_addresses)
                            for p in (pair_ab, pair_bc, pair_ca)
                        ):
                            continue

                        try:
                            hop_types = tuple("v3" if "v3" in p.dex else "v2" for p in (pair_ab, pair_bc, pair_ca))
                            has_v3 = any(h == "v3" for h in hop_types)
                            if has_v3 and not allow_v3:
                                continue

                            # Update reserves (only for V2)
                            for p in [pair_ab, pair_bc, pair_ca]:
                                if "v3" in p.dex:
                                    continue
                                if p.pair_id not in reserve_cache:
                                    r0, r1 = fetch_reserves_with_rotation(rpc_urls, p.pair_id, start_idx)
                                    reserve_cache[p.pair_id] = (r0, r1)
                                    p.reserve0 = to_decimal(r0, p.token0_decimals)
                                    p.reserve1 = to_decimal(r1, p.token1_decimals)

                            estimate_ok = not has_v3
                            profit_safe = Decimal(0)
                            net_profit_usd = None
                            amt_in = Decimal(0)
                            if estimate_ok:
                                amt_in, profit = best_triangle_arb(
                                    start_token, b, c, pair_ab, pair_bc, pair_ca, fee_by_dex, max_trade_frac
                                )
                                if profit <= 0:
                                    continue

                                # Safety haircut
                                safety_mult = (Decimal(10000) - safety_bps) / Decimal(10000)
                                # Profit is (out - in). We apply safety to out.
                                amt_out = amt_in + profit
                                amt_out_safe = amt_out * safety_mult
                                profit_safe = amt_out_safe - amt_in

                                if profit_safe <= 0:
                                    continue

                                price_usd = token_price_usd(start_token, pair_index, rpc_urls, start_idx, weth_price)
                                profit_usd = None if price_usd is None else profit_safe * price_usd
                                if profit_usd is not None and weth_price is not None:
                                    gas_cost_eth = (Decimal(gas_units) * gas_price_gwei) / Decimal(1e9)
                                    gas_cost_usd = gas_cost_eth * weth_price
                                    net_profit_usd = profit_usd - gas_cost_usd

                                if min_net_profit_usd > 0 and (net_profit_usd is None or net_profit_usd < min_net_profit_usd):
                                    continue

                                if min_pair_liquidity_usd > 0:
                                    liq_ab = pair_liquidity_usd(pair_ab, pair_index, rpc_urls, start_idx, weth_price)
                                    liq_bc = pair_liquidity_usd(pair_bc, pair_index, rpc_urls, start_idx, weth_price)
                                    liq_ca = pair_liquidity_usd(pair_ca, pair_index, rpc_urls, start_idx, weth_price)
                                    if (
                                        liq_ab is None
                                        or liq_bc is None
                                        or liq_ca is None
                                        or min(liq_ab, liq_bc, liq_ca) < min_pair_liquidity_usd
                                    ):
                                        continue

                            else:
                                if start_token.lower() != WETH.lower():
                                    continue
                                amt_in = v3_amount_in

                            min_profit_weth = Decimal(0)
                            if (
                                weth_price is not None
                                and min_net_profit_usd > 0
                                and start_token.lower() == WETH.lower()
                            ):
                                min_profit_weth = min_net_profit_usd / weth_price
                            min_profit_weth_raw = int(min_profit_weth * Decimal(10) ** 18) if min_profit_weth > 0 else 0

                            decimals = pair_ab.token0_decimals if pair_ab.token0.lower() == start_token.lower() else pair_ab.token1_decimals
                            raw_amount_in = int(amt_in * (Decimal(10) ** decimals))

                            sim_ok = False
                            if simulate_all or auto_execute:
                                if min_net_profit_usd > 0 and start_token.lower() != WETH.lower():
                                    continue
                                sim_ok = execute_triangular_trade(
                                    start_token, b, c, pair_ab, pair_bc, pair_ca,
                                    hop_types,
                                    raw_amount_in,
                                    rpc_urls[0],
                                    os.getenv("SKIM_PRIVATE_KEY", ""),
                                    gas_price_gwei,
                                    dry_run=True,
                                    monstrosity_address=monstrosity_address,
                                    aave_pool_address=aave_pool_address,
                                    fee_by_dex=fee_by_dex,
                                    safety_bps=safety_bps,
                                    min_profit_weth=min_profit_weth_raw,
                                )
                                if simulate_all:
                                    status = "ok" if sim_ok else "fail"
                                    dex_path = f"{pair_ab.dex}->{pair_bc.dex}->{pair_ca.dex}"
                                    print(
                                        f"SIM {status}: {start_token}->{b}->{c}->{start_token} ({dex_path}) | in={amt_in:.6f}"
                                    )

                            if estimate_ok:
                                net_note = f"net=${net_profit_usd:.2f}" if net_profit_usd is not None else "net=unknown"
                                dex_path = f"{pair_ab.dex}->{pair_bc.dex}->{pair_ca.dex}"
                                print(
                                    f"FOUND TRIANGULAR ARB: {start_token}->{b}->{c}->{start_token} ({dex_path}) | "
                                    f"in={amt_in:.6f} profit={profit_safe:.6f} | {net_note}"
                                )
                            elif sim_ok:
                                dex_path = f"{pair_ab.dex}->{pair_bc.dex}->{pair_ca.dex}"
                                print(
                                    f"FOUND TRIANGULAR ARB (sim-only): {start_token}->{b}->{c}->{start_token} ({dex_path}) | "
                                    f"in={amt_in:.6f}"
                                )

                            if dump_path and (estimate_ok or sim_ok):
                                payload = {
                                    "start_token": start_token,
                                    "token_b": b,
                                    "token_c": c,
                                    "pair_ab": pair_ab.pair_id,
                                    "pair_bc": pair_bc.pair_id,
                                    "pair_ca": pair_ca.pair_id,
                                    "dex_ab": pair_ab.dex,
                                    "dex_bc": pair_bc.dex,
                                    "dex_ca": pair_ca.dex,
                                    "amount_in": str(amt_in),
                                    "amount_in_raw": raw_amount_in,
                                    "start_token_decimals": decimals,
                                    "token_b_decimals": pair_ab.token1_decimals if pair_ab.token0.lower() == start_token.lower() else pair_ab.token0_decimals,
                                    "token_c_decimals": pair_bc.token1_decimals if pair_bc.token0.lower() == b else pair_bc.token0_decimals,
                                    "profit_est": str(profit_safe),
                                    "net_profit_usd_est": None if net_profit_usd is None else str(net_profit_usd),
                                    "fee_bps_ab": int(fee_by_dex.get(pair_ab.dex, Decimal("0.003")) * Decimal(10000)),
                                    "fee_bps_bc": int(fee_by_dex.get(pair_bc.dex, Decimal("0.003")) * Decimal(10000)),
                                    "fee_bps_ca": int(fee_by_dex.get(pair_ca.dex, Decimal("0.003")) * Decimal(10000)),
                                    "safety_bps": int(safety_bps),
                                    "min_profit_weth_raw": min_profit_weth_raw,
                                    "timestamp": int(time.time()),
                                }
                                with open(dump_path, "a", encoding="utf-8") as handle:
                                    handle.write(json.dumps(payload) + "\n")

                            count += 1
                            if sim_ok and auto_execute:
                                if not allow_any and not allow_addresses:
                                    continue
                                execute_triangular_trade(
                                    start_token, b, c, pair_ab, pair_bc, pair_ca,
                                    hop_types,
                                    raw_amount_in,
                                    rpc_urls[0],
                                    os.getenv("SKIM_PRIVATE_KEY", ""),
                                    gas_price_gwei,
                                    dry_run=False,
                                    monstrosity_address=monstrosity_address,
                                    aave_pool_address=aave_pool_address,
                                    fee_by_dex=fee_by_dex,
                                    safety_bps=safety_bps,
                                    min_profit_weth=min_profit_weth_raw,
                                )
                        except Exception as e:
                            # print(f"Error checking cycle: {e}")
                            pass
        if count > 10: break # limit results per iteration


def main() -> None:
    parser = argparse.ArgumentParser(description="Flash swap arbitrage scanner & executor.")
    parser.add_argument("--db-path", default="skim_pairs.db", help="SQLite DB with pairs.")
    parser.add_argument("--max-pairs", type=int, default=0, help="Limit pairs per dex (0 = all).")
    parser.add_argument("--top", type=int, default=25, help="Top opportunities to print.")
    parser.add_argument("--min-profit", type=Decimal, default=Decimal("0"), help="Minimum profit in input token.")
    parser.add_argument("--max-trade-frac", type=Decimal, default=Decimal("1.0"), help="Max trade size as fraction of reserve.") # Aggressive
    parser.add_argument("--fee-uniswap", type=Decimal, default=Decimal("0.003"), help="Uniswap v2 fee.")
    parser.add_argument("--fee-camelot", type=Decimal, default=Decimal("0.005"), help="Camelot v2 fee.")
    parser.add_argument("--fee-sushiswap", type=Decimal, default=Decimal("0.005"), help="Sushi v2 fee.")
    parser.add_argument("--settle-token", choices=["none", "weth", "usdc"], default="none")
    parser.add_argument("--settle-fee-bps", type=Decimal, default=Decimal("30"), help="Fee bps for settle hop.")
    parser.add_argument("--gas-units", type=int, default=500000, help="Gas units for net profit filter.") # Increased for Aave + Steps
    parser.add_argument("--gas-price-gwei", type=Decimal, default=Decimal("0.02"), help="Gas price in gwei.")
    parser.add_argument("--min-net-profit-usd", type=Decimal, default=Decimal("0.01"), help="Minimum net profit in USD.") # Aggressive
    parser.add_argument("--dexes", default="uniswapv2,camelot,sushiswapv2")
    parser.add_argument("--focus-top-reserve", type=int, default=0)
    parser.add_argument("--focus-top-volume", type=int, default=0)
    parser.add_argument("--rpc-urls", default=os.getenv("ARBITRUM_RPC_URLS", ""))
    parser.add_argument("--watchlist", default="")
    parser.add_argument("--loop", action="store_true", help="Run in a continuous loop.")
    parser.add_argument("--auto-execute", action="store_true", help="Simulate and execute profitable opportunities.")
    parser.add_argument("--triangular", action="store_true", help="Scan for triangular arbitrage (A->B->C->A).")
    parser.add_argument("--triangular-auto-execute", action="store_true", help="Execute triangular routes.")
    parser.add_argument("--auto-execute-allow-any", action="store_true", help="Allow auto-execute without allowlist.")
    parser.add_argument("--triangular-safety-bps", type=Decimal, default=Decimal("10"), help="Safety haircut bps.")
    parser.add_argument("--triangular-simulate-all", action="store_true", help="Simulate every found triangular opp.")
    parser.add_argument("--triangular-allow-v3", action="store_true", help="Allow V3 paths (sim-only; no reserve quoting).")
    parser.add_argument("--triangular-v3-amount-in", type=Decimal, default=Decimal("0.1"), help="WETH amount for V3 sim-only paths.")
    parser.add_argument("--complex", action="store_true", help="Scan for complex (multi-hop) arbitrage.")
    parser.add_argument("--min-hops", type=int, default=4, help="Min hops for complex scan.")
    parser.add_argument("--max-hops", type=int, default=6, help="Max hops for complex scan.")
    parser.add_argument("--monstrosity-addr", default=MONSTROSITY_ADDRESS, help="Monstrosity contract address.")
    parser.add_argument("--aave-pool", default=AAVE_V3_POOL_ADDRESS, help="Aave V3 pool address.")
    parser.add_argument("--triangular-dump", default="", help="Append found triangular opps to JSONL file.")
    parser.add_argument("--ignore-addresses", default="", help="Comma-separated addresses to skip.")
    parser.add_argument("--allow-addresses", default="", help="Comma-separated addresses to allow.")
    parser.add_argument("--min-pair-liquidity-usd", type=Decimal, default=Decimal("0"), help="Min USD liquidity per pair.")
    parser.add_argument("--ignore-tokens", default=os.getenv("IGNORE_TOKENS", ""), help="(Deprecated) token addresses to skip.")
    args = parser.parse_args()

    if not args.monstrosity_addr:
        print("monstrosity address required (set --monstrosity-addr or MONSTROSITY_ADDRESS).", file=sys.stderr)
        sys.exit(2)
    if not args.aave_pool:
        print("aave pool address required (set --aave-pool or AAVE_POOL_ADDRESS).", file=sys.stderr)
        sys.exit(2)

    dexes = [d.strip() for d in args.dexes.split(",") if d.strip()]
    dex_aliases = {"sushiswap": "sushiswapv2"}
    dexes = [dex_aliases.get(d, d) for d in dexes]
    ignore_addresses = parse_ignore_addresses(",".join([args.ignore_addresses, args.ignore_tokens]))
    allow_addresses = parse_allow_addresses(args.allow_addresses)

    conn = sqlite3.connect(args.db_path)
    if args.watchlist:
        all_loaded_pairs = load_pairs_from_watchlist(conn, args.watchlist)
        pairs_by_dex: Dict[str, List[PairData]] = {}
        for p in all_loaded_pairs:
            pairs_by_dex.setdefault(p.dex, []).append(p)
    else:
        pairs_by_dex: Dict[str, List[PairData]] = {}
        for dex in dexes:
            pairs_by_dex[dex] = load_pairs(conn, dex, args.max_pairs)

    if ignore_addresses:
        for dex, dex_pairs in list(pairs_by_dex.items()):
            pairs_by_dex[dex] = [
                p
                for p in dex_pairs
                if not (
                    is_ignored_address(p.pair_id, ignore_addresses)
                    or is_ignored_address(p.token0, ignore_addresses)
                    or is_ignored_address(p.token1, ignore_addresses)
                )
            ]
    if allow_addresses:
        for dex, dex_pairs in list(pairs_by_dex.items()):
            pairs_by_dex[dex] = [
                p
                for p in dex_pairs
                if (
                    is_allowed_address(p.pair_id, allow_addresses, allow_any=False)
                    or (
                        is_allowed_address(p.token0, allow_addresses, allow_any=False)
                        and is_allowed_address(p.token1, allow_addresses, allow_any=False)
                    )
                )
            ]

    by_tokens: Dict[Tuple[str, str], Dict[str, PairData]] = {}
    for dex_pairs in pairs_by_dex.values():
        for pair in dex_pairs:
            key = tuple(sorted([pair.token0.lower(), pair.token1.lower()]))
            by_tokens.setdefault(key, {})[pair.dex] = pair

    focus_keys: Optional[set] = None
    if args.focus_top_reserve or args.focus_top_volume:
        if not GRAPH_API_KEY and "gateway.thegraph.com" in UNISWAP_V2_SUBGRAPH:
            print("GRAPH_API_KEY required for focus scan", file=sys.stderr)
        else:
            focus_keys = set()
            subgraphs = {
                "uniswapv2": UNISWAP_V2_SUBGRAPH,
                "camelot": CAMELOT_V2_SUBGRAPH,
                "sushiswapv2": SUSHISWAP_V2_SUBGRAPH,
            }
            for dex in dexes:
                if dex not in subgraphs: continue
                subgraph = subgraphs[dex]
                try:
                    if args.focus_top_reserve:
                        pairs = fetch_top_pairs(subgraph, "reserveUSD", args.focus_top_reserve)
                        focus_keys.update(tuple(sorted(p)) for p in pairs)
                    if args.focus_top_volume:
                        pairs = fetch_top_pairs(subgraph, "volumeUSD", args.focus_top_volume)
                        focus_keys.update(tuple(sorted(p)) for p in pairs)
                except Exception as e:
                    print(f"Graph error {dex}: {e}", file=sys.stderr)

    rpc_urls = build_rpc_pool(args.rpc_urls)
    if not rpc_urls:
        print("no RPC URLs available", file=sys.stderr)
        sys.exit(1)

    fee_by_dex = {
        "uniswapv2": args.fee_uniswap,
        "camelot": args.fee_camelot,
        "sushiswapv2": args.fee_sushiswap,
    }

    all_pairs: List[PairData] = []
    for dex_pairs in pairs_by_dex.values():
        all_pairs.extend(dex_pairs)
    if ignore_addresses:
        all_pairs = [
            p
            for p in all_pairs
            if not (
                is_ignored_address(p.pair_id, ignore_addresses)
                or is_ignored_address(p.token0, ignore_addresses)
                or is_ignored_address(p.token1, ignore_addresses)
            )
        ]
    if allow_addresses:
        all_pairs = [
            p
            for p in all_pairs
            if (
                is_allowed_address(p.pair_id, allow_addresses, allow_any=False)
                or is_allowed_address(p.token0, allow_addresses, allow_any=False)
                or is_allowed_address(p.token1, allow_addresses, allow_any=False)
            )
        ]
    pair_index = build_pair_index(all_pairs)

    iteration = 0
    while True:
        iteration += 1
        print(f"--- Scan Iteration {iteration} ---")
        sys.stdout.flush()
        reserve_cache: Dict[str, Tuple[int, int]] = {}

        weth_price, weth_pair = find_best_price_pair(
            WETH, list(STABLES), pair_index, rpc_urls, iteration
        )
        # Triangular Scan
        if args.triangular:
            triangular_scan(
                all_pairs,
                pair_index,
                rpc_urls,
                iteration,
                reserve_cache,
                weth_price,
                args.gas_units,
                args.gas_price_gwei,
                args.min_net_profit_usd,
                fee_by_dex,
                args.max_trade_frac,
                args.triangular_auto_execute,
                args.triangular_safety_bps,
                args.monstrosity_addr,
                args.aave_pool,
                ignore_addresses,
                args.triangular_dump,
                allow_addresses,
                args.auto_execute_allow_any,
                args.min_pair_liquidity_usd,
                args.triangular_simulate_all,
                args.triangular_allow_v3,
                args.triangular_v3_amount_in,
            )

        # Complex Scan
        if args.complex:
            complex_scan(
                all_pairs,
                pair_index,
                rpc_urls,
                iteration,
                reserve_cache,
                weth_price,
                args.gas_units,
                args.gas_price_gwei,
                args.min_net_profit_usd,
                fee_by_dex,
                args.max_trade_frac,
                args.auto_execute,
                args.triangular_safety_bps,
                args.monstrosity_addr,
                args.aave_pool,
                ignore_addresses,
                args.triangular_dump,
                allow_addresses,
                args.auto_execute_allow_any,
                args.min_pair_liquidity_usd,
                args.triangular_simulate_all,
                args.triangular_allow_v3,
                args.triangular_v3_amount_in,
                args.min_hops,
                args.max_hops,
            )

        keys = [k for k, v in by_tokens.items() if sum(1 for dex in dexes if dex in v) >= 2]
        if focus_keys is not None:
            keys = [k for k in keys if k in focus_keys]

        results = []
        for idx, key in enumerate(keys):
            available = {dex: by_tokens[key][dex] for dex in dexes if dex in by_tokens[key]}

            # Fetch reserves
            bad_dexes = set()
            for dex_idx, (dex, pair) in enumerate(list(available.items())):
                try:
                    if pair.pair_id not in reserve_cache:
                        r0, r1 = fetch_reserves_with_rotation(rpc_urls, pair.pair_id, idx + dex_idx)
                        reserve_cache[pair.pair_id] = (r0, r1)
                    else:
                        r0, r1 = reserve_cache[pair.pair_id]
                    pair.reserve0 = to_decimal(r0, pair.token0_decimals)
                    pair.reserve1 = to_decimal(r1, pair.token1_decimals)
                except Exception:
                    bad_dexes.add(dex)
            for dex in bad_dexes: available.pop(dex, None)
            if len(available) < 2: continue

            candidates = []
            dex_list = list(available.keys())
            for i in range(len(dex_list)):
                for j in range(len(dex_list)):
                    if i == j: continue
                    dex_a = dex_list[i]
                    dex_b = dex_list[j]
                    pair_a = available[dex_a]
                    pair_b = available[dex_b]
                    fee_a = fee_by_dex.get(dex_a, args.fee_uniswap)
                    fee_b = fee_by_dex.get(dex_b, args.fee_uniswap)

                    amt_in0, profit0, token0 = best_arb_for_token0(pair_a, pair_b, fee_a, fee_b, args.max_trade_frac)
                    amt_in1, profit1, token1 = best_arb_for_token1(pair_a, pair_b, fee_a, fee_b, args.max_trade_frac)
                    candidates.append((f"{dex_a}->{dex_b}", amt_in0, profit0, token0, pair_a, pair_b, fee_a, fee_b))
                    candidates.append((f"{dex_a}->{dex_b}", amt_in1, profit1, token1, pair_a, pair_b, fee_a, fee_b))

            best = max(candidates, key=lambda c: c[2])
            if best[2] <= args.min_profit: continue

            _, amount_in, profit, input_token, pair_a, pair_b, fee_a_dec, fee_b_dec = best
            input_token_addr = pair_a.token0 if input_token == "token0" else pair_a.token1
            price_usd = token_price_usd(input_token_addr, pair_index, rpc_urls, idx, weth_price)
            profit_usd = price_usd * profit if price_usd is not None else None

            # Settle logic
            settle_token = args.settle_token
            settle_amount = None
            settle_route = None
            if settle_token != "none":
                target_token = WETH if settle_token == "weth" else "0xaf88d065e77c8cc2239327c5edb3a432268e5831"
                settle_result = settle_profit(
                    input_token_addr, profit, target_token, pair_index, rpc_urls, idx, reserve_cache, args.settle_fee_bps
                )
                if settle_result:
                    settle_amount, settle_route = settle_result
                    if settle_amount is None:
                        continue
                    if settle_token == "usdc":
                        profit_usd = settle_amount
                    elif settle_token == "weth" and weth_price is not None:
                        profit_usd = settle_amount * weth_price

            net_profit_usd = None
            if profit_usd is not None and weth_price is not None:
                gas_cost_eth = (Decimal(args.gas_units) * args.gas_price_gwei) / Decimal(1e9)
                gas_cost_usd = gas_cost_eth * weth_price
                net_profit_usd = profit_usd - gas_cost_usd

            if args.min_net_profit_usd > 0 and (net_profit_usd is None or net_profit_usd < args.min_net_profit_usd):
                continue

            opportunity = {
                "pair_key": key,
                "route": best[0],
                "input_token": input_token,
                "amount_in": amount_in,
                "profit": profit,
                "profit_usd": profit_usd,
                "net_profit_usd": net_profit_usd,
                "settle_token": settle_token,
                "settle_amount": settle_amount,
                "settle_route": settle_route,
                "pair_a": pair_a,
                "pair_b": pair_b,
                "fee_a_bps": int(fee_a_dec * 10000),
                "fee_b_bps": int(fee_b_dec * 10000)
            }
            results.append(opportunity)

            if args.auto_execute and net_profit_usd and net_profit_usd > 0:
                print(f"Found profitable arb: {opportunity['route']} profit=${profit_usd:.2f} net=${net_profit_usd:.2f}")
                raw_amount_in = int(amount_in * (Decimal(10) ** (pair_a.token0_decimals if input_token == "token0" else pair_a.token1_decimals)))

                success = execute_trade(
                    pair_borrow=pair_b.pair_id,
                    pair_swap=pair_a.pair_id,
                    token_borrow=input_token_addr,
                    amount_borrow=raw_amount_in,
                    fee_borrow_bps=opportunity["fee_b_bps"],
                    fee_swap_bps=opportunity["fee_a_bps"],
                    min_profit=0,
                    dry_run=False,
                    rpc_url=rpc_urls[0],
                    private_key=os.getenv("SKIM_PRIVATE_KEY", ""),
                    gas_price_gwei=args.gas_price_gwei,
                    monstrosity_address=args.monstrosity_addr,
                    aave_pool_address=args.aave_pool,
                )
                if success:
                    print("Execution Success!")
                else:
                    print("Execution Failed.")

        results.sort(key=lambda r: r["profit"], reverse=True)
        if not results:
            print("no results found")
        else:
            for item in results[: args.top]:
                print(f"{item['route']} profit={item['profit']} (${item['profit_usd']}) net=${item['net_profit_usd']}")

        if not args.loop:
            break
        time.sleep(1)

if __name__ == "__main__":
    main()

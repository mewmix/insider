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
from web3.exceptions import ContractLogicError

from skim_scanner import RPC_ENDPOINTS, normalize_rpc_url
from ignore_list import parse_ignore_addresses, is_ignored_address
from policy import parse_allow_addresses, is_allowed_address

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

FLASH_ARB_ADDRESS = "0xe14b184315f0a1edc476032daa051d7e6465858b"
FLASH_ARB_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "pairBorrow", "type": "address"},
            {"internalType": "address", "name": "pairSwap", "type": "address"},
            {"internalType": "address", "name": "tokenBorrow", "type": "address"},
            {"internalType": "uint256", "name": "amountBorrow", "type": "uint256"},
            {"internalType": "uint256", "name": "feeBorrowBps", "type": "uint256"},
            {"internalType": "uint256", "name": "feeSwapBps", "type": "uint256"},
            {"internalType": "uint256", "name": "minProfit", "type": "uint256"},
        ],
        "name": "execute",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]


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
) -> bool:
    if not private_key:
        print("skipping execution: no private key", file=sys.stderr)
        return False

    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        print("execution error: RPC not connected", file=sys.stderr)
        return False

    account = w3.eth.account.from_key(private_key)
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(FLASH_ARB_ADDRESS),
        abi=FLASH_ARB_ABI
    )

    pair_borrow_c = Web3.to_checksum_address(pair_borrow)
    pair_swap_c = Web3.to_checksum_address(pair_swap)
    token_borrow_c = Web3.to_checksum_address(token_borrow)

    print(f"Simulating {amount_borrow} of {token_borrow} on {pair_borrow} -> {pair_swap}")
    sys.stdout.flush()

    tx_func = contract.functions.execute(
        pair_borrow_c,
        pair_swap_c,
        token_borrow_c,
        amount_borrow,
        fee_borrow_bps,
        fee_swap_bps,
        min_profit
    )

    try:
        gas_estimate = tx_func.estimate_gas({"from": account.address})
        print(f"Simulation success! Gas: {gas_estimate}")
        sys.stdout.flush()
    except ContractLogicError as e:
        print(f"Simulation failed (revert): {e}", file=sys.stderr)
        sys.stderr.flush()
        return False
    except Exception as e:
        print(f"Simulation failed: {e}", file=sys.stderr)
        sys.stderr.flush()
        return False

    if dry_run:
        return True

    print("Executing transaction...")
    nonce = w3.eth.get_transaction_count(account.address)
    latest_block = w3.eth.get_block("latest")
    base_fee = latest_block.get("baseFeePerGas")

    tx_params = {
        "from": account.address,
        "nonce": nonce,
        "chainId": 42161,
        "gas": int(gas_estimate * 1.2),
    }

    if base_fee is not None:
        max_priority_fee = w3.to_wei(0.1, "gwei")
        max_fee = int(base_fee * 1.35 + max_priority_fee)
        tx_params["maxPriorityFeePerGas"] = max_priority_fee
        tx_params["maxFeePerGas"] = max_fee
        tx_params["type"] = 2
    else:
        tx_params["gasPrice"] = int(w3.eth.gas_price * 1.1)

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


def triangular_dry_run(
    start_token: str,
    token_b: str,
    token_c: str,
    pair_ab: PairData,
    pair_bc: PairData,
    pair_ca: PairData,
    pair_index: Dict[Tuple[str, str], List[PairData]],
    rpc_urls: List[str],
    start_idx: int,
    fee_by_dex: Dict[str, Decimal],
    max_trade_frac: Decimal,
    weth_price: Optional[Decimal],
    gas_units: int,
    gas_price_gwei: Decimal,
    min_net_profit_usd: Decimal,
    safety_bps: Decimal,
) -> bool:
    for p in [pair_ab, pair_bc, pair_ca]:
        r0, r1 = fetch_reserves_with_rotation(rpc_urls, p.pair_id, start_idx)
        p.reserve0 = to_decimal(r0, p.token0_decimals)
        p.reserve1 = to_decimal(r1, p.token1_decimals)

    amt_in, profit = best_triangle_arb(
        start_token, token_b, token_c, pair_ab, pair_bc, pair_ca, fee_by_dex, max_trade_frac
    )
    if profit <= 0:
        print("Triangular dry-run failed: no profit after recheck")
        return False

    amt_out = amt_in + profit
    safety_mult = (Decimal(10000) - safety_bps) / Decimal(10000)
    amt_out_safe = amt_out * safety_mult
    profit_safe = amt_out_safe - amt_in
    if profit_safe <= 0:
        print("Triangular dry-run failed: safety haircut removed profit")
        return False

    price_usd = token_price_usd(start_token, pair_index, rpc_urls, start_idx, weth_price)
    profit_usd = None if price_usd is None else profit_safe * price_usd
    net_profit_usd = None
    if profit_usd is not None and weth_price is not None:
        gas_cost_eth = (Decimal(gas_units) * gas_price_gwei) / Decimal(1e9)
        gas_cost_usd = gas_cost_eth * weth_price
        net_profit_usd = profit_usd - gas_cost_usd

    if min_net_profit_usd > 0 and (net_profit_usd is None or net_profit_usd < min_net_profit_usd):
        print("Triangular dry-run failed: net profit below threshold")
        return False

    net_note = f"net=${net_profit_usd:.2f}" if net_profit_usd is not None else "net=unknown"
    print(
        f"TRIANGULAR DRY RUN PASS: {start_token}->{token_b}->{token_c}->{start_token} "
        f"| in={amt_in:.6f} out={amt_out_safe:.6f} | {net_note}"
    )
    return True


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
    ignore_addresses: Set[str],
) -> None:
    # Basic graph: token -> neighbor_token -> pair
    adj: Dict[str, List[Tuple[str, PairData]]] = {}
    for p in pairs:
        t0 = p.token0.lower()
        t1 = p.token1.lower()
        adj.setdefault(t0, []).append((t1, p))
        adj.setdefault(t1, []).append((t0, p))

    # DFS for cycles of length 3: A -> B -> C -> A
    # Limit: Check top 50 tokens by connectivity to avoid explosion
    sorted_tokens = sorted(adj.keys(), key=lambda k: len(adj[k]), reverse=True)

    print(f"Scanning triangular arb opportunities (Graph size: {len(adj)} tokens)...")

    count = 0

    # We will only check the most connected tokens to save time
    for start_token in sorted_tokens[:100]:
        if is_ignored_address(start_token, ignore_addresses):
            continue
        for (b, pair_ab) in adj[start_token]:
            if b == start_token: continue
            if is_ignored_address(b, ignore_addresses):
                continue
            for (c, pair_bc) in adj[b]:
                if c == start_token or c == b: continue
                if is_ignored_address(c, ignore_addresses):
                    continue
                for (d, pair_ca) in adj[c]:
                    if d == start_token:
                        if any(
                            is_ignored_address(p.pair_id, ignore_addresses)
                            for p in (pair_ab, pair_bc, pair_ca)
                        ):
                            continue
                        # Found cycle A -> B -> C -> A
                        # Verify we have pairs: pair_ab, pair_bc, pair_ca
                        # Calculate profit...
                        # Since we can't execute, we just verify reserves and print
                        try:
                            # Update reserves if needed
                            for p in [pair_ab, pair_bc, pair_ca]:
                                if p.pair_id not in reserve_cache:
                                    r0, r1 = fetch_reserves_with_rotation(rpc_urls, p.pair_id, start_idx)
                                    reserve_cache[p.pair_id] = (r0, r1)
                                    p.reserve0 = to_decimal(r0, p.token0_decimals)
                                    p.reserve1 = to_decimal(r1, p.token1_decimals)

                            amt_in, profit = best_triangle_arb(
                                start_token, b, c, pair_ab, pair_bc, pair_ca, fee_by_dex, max_trade_frac
                            )
                            if profit <= 0:
                                continue

                            amt_out = amt_in + profit
                            price_usd = token_price_usd(start_token, pair_index, rpc_urls, start_idx, weth_price)
                            profit_usd = None if price_usd is None else profit * price_usd
                            net_profit_usd = None
                            if profit_usd is not None and weth_price is not None:
                                gas_cost_eth = (Decimal(gas_units) * gas_price_gwei) / Decimal(1e9)
                                gas_cost_usd = gas_cost_eth * weth_price
                                net_profit_usd = profit_usd - gas_cost_usd

                            if min_net_profit_usd > 0 and (net_profit_usd is None or net_profit_usd < min_net_profit_usd):
                                continue

                            net_note = f"net=${net_profit_usd:.2f}" if net_profit_usd is not None else "net=unknown"
                            print(
                                f"FOUND TRIANGULAR ARB: {start_token}->{b}->{c}->{start_token} | "
                                f"in={amt_in:.6f} out={amt_out:.6f} | {net_note} | (Cannot execute with current contract)"
                            )
                            count += 1
                            if auto_execute:
                                triangular_dry_run(
                                    start_token,
                                    b,
                                    c,
                                    pair_ab,
                                    pair_bc,
                                    pair_ca,
                                    pair_index,
                                    rpc_urls,
                                    start_idx + 1,
                                    fee_by_dex,
                                    max_trade_frac,
                                    weth_price,
                                    gas_units,
                                    gas_price_gwei,
                                    min_net_profit_usd,
                                    safety_bps,
                                )
                        except Exception:
                            pass
        if count > 5: break # limit results


def main() -> None:
    parser = argparse.ArgumentParser(description="Flash swap arbitrage scanner & executor.")
    parser.add_argument("--db-path", default="skim_pairs.db", help="SQLite DB with pairs.")
    parser.add_argument("--max-pairs", type=int, default=0, help="Limit pairs per dex (0 = all).")
    parser.add_argument("--top", type=int, default=25, help="Top opportunities to print.")
    parser.add_argument("--min-profit", type=Decimal, default=Decimal("0"), help="Minimum profit in input token.")
    parser.add_argument("--max-trade-frac", type=Decimal, default=Decimal("0.3"), help="Max trade size as fraction of reserve.")
    parser.add_argument("--fee-uniswap", type=Decimal, default=Decimal("0.003"), help="Uniswap v2 fee.")
    parser.add_argument("--fee-camelot", type=Decimal, default=Decimal("0.005"), help="Camelot v2 fee.")
    parser.add_argument("--fee-sushiswap", type=Decimal, default=Decimal("0.005"), help="Sushi v2 fee.")
    parser.add_argument("--settle-token", choices=["none", "weth", "usdc"], default="none")
    parser.add_argument("--settle-fee-bps", type=Decimal, default=Decimal("30"), help="Fee bps for settle hop.")
    parser.add_argument("--gas-units", type=int, default=200000, help="Gas units for net profit filter.")
    parser.add_argument("--gas-price-gwei", type=Decimal, default=Decimal("0.02"), help="Gas price in gwei.")
    parser.add_argument("--min-net-profit-usd", type=Decimal, default=Decimal("1.00"), help="Minimum net profit in USD.")
    parser.add_argument("--dexes", default="uniswapv2,camelot,sushiswapv2")
    parser.add_argument("--focus-top-reserve", type=int, default=0)
    parser.add_argument("--focus-top-volume", type=int, default=0)
    parser.add_argument("--rpc-urls", default=os.getenv("ARBITRUM_RPC_URLS", ""))
    parser.add_argument("--watchlist", default="")
    parser.add_argument("--loop", action="store_true", help="Run in a continuous loop.")
    parser.add_argument("--auto-execute", action="store_true", help="Simulate and execute profitable opportunities.")
    parser.add_argument("--auto-execute-allow-any", action="store_true", help="Allow auto-execute without allowlist.")
    parser.add_argument("--auto-execute-dry-run-only", action="store_true", help="Only simulate; do not execute.")
    parser.add_argument("--triangular", action="store_true", help="Scan for triangular arbitrage (A->B->C->A).")
    parser.add_argument("--triangular-auto-execute", action="store_true", help="Dry-run simulate triangular routes after discovery.")
    parser.add_argument("--triangular-safety-bps", type=Decimal, default=Decimal("50"), help="Safety haircut bps for triangular dry-run.")
    parser.add_argument("--ignore-addresses", default="", help="Comma-separated addresses to ignore.")
    parser.add_argument("--allow-addresses", default="", help="Comma-separated addresses to allow.")
    args = parser.parse_args()

    dexes = [d.strip() for d in args.dexes.split(",") if d.strip()]
    dex_aliases = {"sushiswap": "sushiswapv2"}
    dexes = [dex_aliases.get(d, d) for d in dexes]
    ignore_addresses = parse_ignore_addresses(args.ignore_addresses)
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
                or (
                    is_allowed_address(p.token0, allow_addresses, allow_any=False)
                    and is_allowed_address(p.token1, allow_addresses, allow_any=False)
                )
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
                ignore_addresses,
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
                if not args.auto_execute_allow_any and not allow_addresses:
                    continue
                if allow_addresses and not all(
                    is_allowed_address(addr, allow_addresses, allow_any=False)
                    for addr in (pair_a.pair_id, pair_b.pair_id, input_token_addr)
                ):
                    continue
                print(f"Found profitable arb: {opportunity['route']} profit=${profit_usd:.2f} net=${net_profit_usd:.2f}")
                # Format args for execution
                # We need raw integer amounts
                raw_amount_in = int(amount_in * (Decimal(10) ** (pair_a.token0_decimals if input_token == "token0" else pair_a.token1_decimals)))
                # Calculate raw min profit? set to 0 to ensure execution for now, or strict
                # Using 0 allows minor slippage.

                # BUG FIX: If Python calculated A->B, we must borrow from B to execute A->B via flash loan.
                # execute(pairBorrow=B, pairSwap=A, ...)
                # Because execution logic is: Borrow(B) -> Swap(A) -> Swap(B, repay)
                # This corresponds to route: A -> B

                success = execute_trade(
                    pair_borrow=pair_b.pair_id,  # Was pair_a.pair_id
                    pair_swap=pair_a.pair_id,    # Was pair_b.pair_id
                    token_borrow=input_token_addr,
                    amount_borrow=raw_amount_in,
                    fee_borrow_bps=opportunity["fee_b_bps"], # Was fee_a_bps
                    fee_swap_bps=opportunity["fee_a_bps"],   # Was fee_b_bps
                    min_profit=0,
                    dry_run=True,
                    rpc_url=rpc_urls[0],
                    private_key=os.getenv("SKIM_PRIVATE_KEY", "")
                )
                if success and not args.auto_execute_dry_run_only:
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
                        private_key=os.getenv("SKIM_PRIVATE_KEY", "")
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

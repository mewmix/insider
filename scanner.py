import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Tuple, Set, Any

import aiohttp
from dotenv import load_dotenv
from web3 import AsyncWeb3, AsyncHTTPProvider, Web3
from eth_abi import encode as eth_abi_encode
from web3.exceptions import ContractLogicError

from ignore_list import parse_ignore_addresses, is_ignored_address
from policy import parse_allow_addresses, is_allowed_address
from scanner_config import RPC_ENDPOINTS, normalize_rpc_url

load_dotenv()
getcontext().prec = 60

# --- Configuration & Constants ---

UNISWAP_V2_SUBGRAPH = os.getenv(
    "UNISWAP_V2_SUBGRAPH",
    "https://gateway.thegraph.com/api/subgraphs/id/CStW6CSQbHoXsgKuVCrk3uShGA4JX3CAzzv2x9zaGf8w",
)

MONSTROSITY_ADDRESS = os.getenv(
    "MONSTROSITY_ADDRESS",
    "0x7e5E849D5a3FBAea7044b4b9e47baBb3d6A60283",
)
AAVE_V3_POOL_ADDRESS = os.getenv(
    "AAVE_POOL_ADDRESS",
    "0x794a61358D6845594F94dc1DB02A252b5b4814aD",
)

WETH = "0x82af49447d8a07e3bd95bd0d56f35241523fbab1"
STABLES = {
    "0xff970a61a04b1ca14834a43f5de4533ebddb5cc8",  # USDC.e
    "0xaf88d065e77c8cc2239327c5edb3a432268e5831",  # USDC
    "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9",  # USDT
    "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1",  # DAI
}

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

GET_RESERVES_SIG = "0902f1ac"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AsyncScanner")

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

def to_decimal(raw: int, decimals: int) -> Decimal:
    return Decimal(raw) / (Decimal(10) ** decimals)

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

# --- Async RPC Manager ---

class AsyncRPCManager:
    def __init__(self, rpc_urls: List[str]):
        self.rpc_urls = rpc_urls
        self.index = 0
        self.session: Optional[aiohttp.ClientSession] = None
        self.working_rpcs = list(rpc_urls) # Start with all

    async def get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            conn = aiohttp.TCPConnector(limit=100, ttl_dns_cache=300)
            self.session = aiohttp.ClientSession(connector=conn)
        return self.session

    async def call(self, method: str, params: List[Any], request_id: int = 1) -> Any:
        session = await self.get_session()
        attempts = len(self.working_rpcs)
        if attempts == 0:
            self.working_rpcs = list(self.rpc_urls)
            attempts = len(self.working_rpcs)

        last_error = None
        start_index = (self.index) % len(self.working_rpcs)
        self.index += 1

        for i in range(attempts):
            idx = (start_index + i) % len(self.working_rpcs)
            url = self.working_rpcs[idx]

            try:
                async with session.post(
                    url,
                    json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
                    timeout=10
                ) as resp:
                    if resp.status == 429:
                        continue
                    resp.raise_for_status()
                    payload = await resp.json()
                    if "error" in payload:
                        raise RuntimeError(payload["error"])
                    return payload["result"]
            except Exception as e:
                last_error = e
                continue

        # If all fail, try full list once just in case
        raise RuntimeError(f"All RPCs failed. Last error: {last_error}")

    async def close(self):
        if self.session:
            await self.session.close()

async def fetch_reserves_raw(manager: AsyncRPCManager, pair: str) -> Tuple[int, int]:
    try:
        result = await manager.call("eth_call", [{"to": pair, "data": "0x" + GET_RESERVES_SIG}, "latest"])
        raw = result[2:]
        if len(raw) < 128:
             return 0, 0
        return int(raw[0:64], 16), int(raw[64:128], 16)
    except Exception:
        return 0, 0

async def fetch_pair_reserves(manager: AsyncRPCManager, pair: PairData) -> None:
    if "v3" in pair.dex:
        pair.reserve0 = Decimal("1000000")
        pair.reserve1 = Decimal("1000000")
        return

    r0, r1 = await fetch_reserves_raw(manager, pair.pair_id)
    pair.reserve0 = to_decimal(r0, pair.token0_decimals)
    pair.reserve1 = to_decimal(r1, pair.token1_decimals)

# --- DB & Data Loading (Sync) ---

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

# --- Calculation Logic (Sync) ---

def swap_out(amount_in: Decimal, reserve_in: Decimal, reserve_out: Decimal, fee: Decimal) -> Decimal:
    if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
        return Decimal(0)
    amount_in_with_fee = amount_in * (Decimal(1) - fee)
    return (amount_in_with_fee * reserve_out) / (reserve_in + amount_in_with_fee)

def swap_out_pair(amount_in: Decimal, token_in: str, pair: PairData, fee: Decimal) -> Decimal:
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
    if pool_a.reserve0 <= 0: return Decimal(0), Decimal(0), "token0"
    max_in = pool_a.reserve0 * max_trade_frac

    def profit_fn(amount_in: Decimal) -> Decimal:
        out_1 = swap_out(amount_in, pool_a.reserve0, pool_a.reserve1, fee_a)
        out_2 = swap_out(out_1, pool_b.reserve1, pool_b.reserve0, fee_b)
        return out_2 - amount_in

    amt, profit = ternary_search_generic(max_in, profit_fn)
    return amt, profit, "token0"

def best_arb_for_token1(
    pool_a: PairData,
    pool_b: PairData,
    fee_a: Decimal,
    fee_b: Decimal,
    max_trade_frac: Decimal,
) -> Tuple[Decimal, Decimal, str]:
    if pool_a.reserve1 <= 0: return Decimal(0), Decimal(0), "token1"
    max_in = pool_a.reserve1 * max_trade_frac

    def profit_fn(amount_in: Decimal) -> Decimal:
        out_1 = swap_out(amount_in, pool_a.reserve1, pool_a.reserve0, fee_a)
        out_2 = swap_out(out_1, pool_b.reserve0, pool_b.reserve1, fee_b)
        return out_2 - amount_in

    amt, profit = ternary_search_generic(max_in, profit_fn)
    return amt, profit, "token1"


# --- Async Execution ---

async def execute_monstrosity_async(
    w3: AsyncWeb3,
    steps: List[Dict],
    contract_address: str,
    private_key: str,
    gas_price_gwei: Decimal,
    min_profit_weth: int,
    dry_run: bool = False,
) -> bool:
    account = w3.eth.account.from_key(private_key)
    contract = w3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=MONSTROSITY_ABI)

    logger.info(f"Executing with {len(steps)} top-level steps...")

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
        gas_estimate = await tx_func.estimate_gas({"from": account.address})
        logger.info(f"Simulation success! Gas: {gas_estimate}")
    except ContractLogicError as e:
        logger.error(f"Simulation failed (revert): {e}")
        return False
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return False

    if dry_run:
        return True

    logger.info("Submitting transaction...")
    nonce = await w3.eth.get_transaction_count(account.address)

    tx_params = {
        "from": account.address,
        "nonce": nonce,
        "chainId": 42161,
        "gas": int(gas_estimate * 1.2),
        "gasPrice": int(gas_price_gwei * Decimal(1e9))
    }

    try:
        # Build transaction
        tx = await tx_func.build_transaction(tx_params)
        # Sign locally (cpu bound, fast enough)
        signed_tx = w3.eth.account.sign_transaction(tx, private_key)
        # Send
        tx_hash = await w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        logger.info(f"Transaction sent: {tx_hash.hex()}")

        receipt = await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if receipt.status == 1:
            logger.info("Transaction confirmed!")
            return True
        else:
            logger.error("Transaction reverted on chain.")
            return False
    except Exception as e:
        logger.error(f"Execution error: {e}")
        return False

async def execute_trade_async(
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
        logger.error("skipping execution: no private key")
        return False

    w3 = AsyncWeb3(AsyncHTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
    if not await w3.is_connected():
        return False

    pair_1 = pair_swap
    pair_2 = pair_borrow

    step1 = {
        "action": ACTION_V2_SWAP,
        "target": pair_1,
        "tokenIn": token_borrow,
        "tokenOut": "0x0000000000000000000000000000000000000000",
        "amountIn": amount_borrow,
        "minAmountOut": 0,
        "extraData": b""
    }
    if fee_swap_bps != 30:
        step1["extraData"] = eth_abi_encode(["uint256"], [fee_swap_bps])

    # We need to find token_intermediate to correctly format step2
    # Pair 1 has token0/token1. One is token_borrow.
    pair1_contract = w3.eth.contract(address=Web3.to_checksum_address(pair_1), abi=[
        {"constant":True,"inputs":[],"name":"token0","outputs":[{"name":"","type":"address"}],"type":"function"},
        {"constant":True,"inputs":[],"name":"token1","outputs":[{"name":"","type":"address"}],"type":"function"}
    ])
    p1_t0 = await pair1_contract.functions.token0().call()
    # p1_t1 = await pair1_contract.functions.token1().call()

    token_intermediate = (await pair1_contract.functions.token1().call()) if p1_t0.lower() == token_borrow.lower() else p1_t0

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

    nested_steps = [step1, step2]
    step_type = "(uint8,address,address,address,uint256,uint256,bytes)"
    encoded_nested = eth_abi_encode([f"{step_type}[]"], [ [
        (s["action"], s["target"], s["tokenIn"], s["tokenOut"], s["amountIn"], s["minAmountOut"], s["extraData"])
        for s in nested_steps
    ] ])

    flash_step = {
        "action": ACTION_AAVE_FLASH,
        "target": aave_pool_address,
        "tokenIn": token_borrow,
        "tokenOut": token_borrow,
        "amountIn": amount_borrow,
        "minAmountOut": 0,
        "extraData": encoded_nested
    }

    return await execute_monstrosity_async(
        w3, [flash_step], monstrosity_address, private_key, gas_price_gwei, 0, dry_run
    )

async def execute_triangular_trade_async(
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
    w3 = AsyncWeb3(AsyncHTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
    if not await w3.is_connected(): return False

    fee_ab = fee_by_dex.get(pair_ab.dex, Decimal("0.003"))
    fee_bc = fee_by_dex.get(pair_bc.dex, Decimal("0.003"))
    fee_ca = fee_by_dex.get(pair_ca.dex, Decimal("0.003"))

    # We assume reserves are reasonably fresh or we accept simulation failure if not.
    # To be safe in execution, we should probably refetch or just trust simulation.
    # The sync code refetched. We will just proceed with logic.

    # Recalculate min outs based on provided amount_in
    in_decimals = pair_ab.token0_decimals if pair_ab.token0.lower() == start_token.lower() else pair_ab.token1_decimals
    amount_in_dec = to_decimal(amount_in, in_decimals)

    # We need reserves for estimation. We can fetch them via web3 or assume PairData has them?
    # PairData might be stale.
    # But for execution, the important part is constructing the tx.
    # We will use PairData passed in.

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

    return await execute_monstrosity_async(
        w3, [flash_step], monstrosity_address, private_key, gas_price_gwei, min_profit_weth, dry_run
    )

async def execute_path_trade_async(
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
    if not private_key: return False
    w3 = AsyncWeb3(AsyncHTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
    if not await w3.is_connected(): return False

    start_token = path_tokens[0]
    in_decimals = (
        path_pairs[0].token0_decimals
        if path_pairs[0].token0.lower() == start_token.lower()
        else path_pairs[0].token1_decimals
    )
    amount_in_dec = to_decimal(amount_in, in_decimals)

    # Note: We rely on PairData having correct decimals. Reserves not needed for step construction except for minAmountOut if we calculate it.
    # We will assume caller provided valid pairs.

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

    return await execute_monstrosity_async(
        w3, [flash_step], monstrosity_address, private_key, gas_price_gwei, min_profit_weth, dry_run
    )

# --- Scanning Tasks ---

async def triangular_scan_async(
    manager: AsyncRPCManager,
    pairs: List[PairData],
    pair_index: Dict[Tuple[str, str], List[PairData]],
    dump_path: str,
    args: argparse.Namespace,
    fee_by_dex: Dict[str, Decimal],
    reserve_cache: Dict[str, Tuple[int, int]],
):
    adj: Dict[str, List[Tuple[str, PairData]]] = {}
    ignore_addresses = parse_ignore_addresses(args.ignore_addresses)
    allow_addresses = parse_allow_addresses(args.allow_addresses)

    for p in pairs:
        t0 = p.token0.lower()
        t1 = p.token1.lower()
        if allow_addresses:
            if not (is_allowed_address(t0, allow_addresses, allow_any=False) or is_allowed_address(t1, allow_addresses, allow_any=False)):
                continue
        adj.setdefault(t0, []).append((t1, p))
        adj.setdefault(t1, []).append((t0, p))

    sorted_tokens = sorted(adj.keys(), key=lambda k: len(adj[k]), reverse=True)
    logger.info(f"Scanning triangular arb (Graph size: {len(adj)} tokens)...")

    scan_tokens = sorted_tokens[:300]
    tasks = []
    sem = asyncio.Semaphore(50)

    async def check_cycle(start_token: str):
        async with sem:
            if start_token in ignore_addresses: return
            neighbors_b = adj.get(start_token, [])
            for (b, pair_ab) in neighbors_b:
                if b == start_token or b in ignore_addresses: continue
                neighbors_c = adj.get(b, [])
                for (c, pair_bc) in neighbors_c:
                    if c == start_token or c == b or c in ignore_addresses: continue
                    neighbors_d = adj.get(c, [])
                    for (d, pair_ca) in neighbors_d:
                        if d == start_token:
                             # Found cycle start->b->c->start
                            if any(tok in ignore_addresses for tok in (start_token, b, c)): continue

                            # Parallel fetch reserves
                            to_fetch = []
                            for p in [pair_ab, pair_bc, pair_ca]:
                                if "v3" not in p.dex and p.pair_id not in reserve_cache:
                                    to_fetch.append(p)

                            if to_fetch:
                                await asyncio.gather(*(fetch_pair_reserves(manager, p) for p in to_fetch))
                                for p in to_fetch:
                                    reserve_cache[p.pair_id] = (0,0) # dummy, values in object updated

                            hop_types = tuple("v3" if "v3" in p.dex else "v2" for p in (pair_ab, pair_bc, pair_ca))
                            has_v3 = any(h == "v3" for h in hop_types)

                            if has_v3 and not args.triangular_allow_v3: continue

                            amt_in = Decimal(0)
                            profit_safe = Decimal(0)

                            if not has_v3:
                                amt_in, profit = best_triangle_arb(
                                    start_token, b, c, pair_ab, pair_bc, pair_ca, fee_by_dex, args.max_trade_frac
                                )
                                if profit <= 0: continue
                                safety_mult = (Decimal(10000) - args.triangular_safety_bps) / Decimal(10000)
                                amt_out_safe = (amt_in + profit) * safety_mult
                                profit_safe = amt_out_safe - amt_in
                            else:
                                if start_token.lower() != WETH.lower(): continue
                                amt_in = args.triangular_v3_amount_in

                            if not has_v3 and profit_safe <= 0: continue

                            # Log found
                            decimals = pair_ab.token0_decimals if pair_ab.token0.lower() == start_token.lower() else pair_ab.token1_decimals

                            # Dump
                            if dump_path:
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
                                    "amount_in_raw": int(amt_in * (Decimal(10) ** decimals)),
                                    "profit_est": str(profit_safe),
                                    "timestamp": int(time.time()),
                                }
                                with open(dump_path, "a", encoding="utf-8") as handle:
                                    handle.write(json.dumps(payload) + "\n")

                            # Execute
                            if args.auto_execute or args.triangular_auto_execute:
                                if not args.auto_execute_allow_any and not allow_addresses: continue

                                raw_amount_in = int(amt_in * (Decimal(10) ** decimals))
                                logger.info(f"Auto-executing triangular arb: {start_token}->{b}->{c}")

                                await execute_triangular_trade_async(
                                    start_token, b, c, pair_ab, pair_bc, pair_ca,
                                    hop_types,
                                    raw_amount_in,
                                    manager.rpc_urls[0],
                                    os.getenv("SKIM_PRIVATE_KEY", ""),
                                    args.gas_price_gwei,
                                    False, # dry_run
                                    MONSTROSITY_ADDRESS,
                                    AAVE_V3_POOL_ADDRESS,
                                    fee_by_dex,
                                    args.triangular_safety_bps,
                                    0
                                )

    for t in scan_tokens:
        tasks.append(asyncio.create_task(check_cycle(t)))

    await asyncio.gather(*tasks)

async def narrow_scan_async(
    manager: AsyncRPCManager,
    keys: List[Tuple[str, str]],
    by_tokens: Dict[Tuple[str, str], Dict[str, PairData]],
    args: argparse.Namespace,
    fee_by_dex: Dict[str, Decimal],
):
    # Flatten all pairs needed
    needed_pairs = set()
    for key in keys:
        for dex, pair in by_tokens[key].items():
            needed_pairs.add(pair)

    # Batch fetch reserves
    logger.info(f"Narrow Scan: Fetching reserves for {len(needed_pairs)} pairs...")
    await asyncio.gather(*(fetch_pair_reserves(manager, p) for p in needed_pairs))

    results = []

    for key in keys:
        available = by_tokens[key]
        dex_list = list(available.keys())
        if len(dex_list) < 2: continue

        candidates = []
        for i in range(len(dex_list)):
            for j in range(len(dex_list)):
                if i == j: continue
                dex_a = dex_list[i]
                dex_b = dex_list[j]
                pair_a = available[dex_a]
                pair_b = available[dex_b]
                fee_a = fee_by_dex.get(dex_a, args.fee_uniswap)
                fee_b = fee_by_dex.get(dex_b, args.fee_uniswap)

                # Check V2 only support for now (ternary search logic is for V2 curves)
                if "v3" in dex_a or "v3" in dex_b: continue

                amt_in0, profit0, token0 = best_arb_for_token0(pair_a, pair_b, fee_a, fee_b, args.max_trade_frac)
                amt_in1, profit1, token1 = best_arb_for_token1(pair_a, pair_b, fee_a, fee_b, args.max_trade_frac)

                candidates.append((f"{dex_a}->{dex_b}", amt_in0, profit0, token0, pair_a, pair_b, fee_a, fee_b))
                candidates.append((f"{dex_a}->{dex_b}", amt_in1, profit1, token1, pair_a, pair_b, fee_a, fee_b))

        if not candidates: continue
        best = max(candidates, key=lambda c: c[2])
        if best[2] <= args.min_profit: continue

        route, amount_in, profit, input_token, pair_a, pair_b, fee_a_dec, fee_b_dec = best

        # Simple profit check (skipping USD pricing for speed for now, relying on min_profit token amount)
        # Or simplistic: check if token is stable/WETH?
        # User requested rapid iteration.

        logger.info(f"Narrow Arb Found: {route} Profit: {profit} ({input_token})")
        results.append(best)

        if args.auto_execute:
             input_token_addr = pair_a.token0 if input_token == "token0" else pair_a.token1
             raw_amount_in = int(amount_in * (Decimal(10) ** (pair_a.token0_decimals if input_token == "token0" else pair_a.token1_decimals)))

             logger.info(f"Executing Narrow Arb: {route}")
             await execute_trade_async(
                 pair_borrow=pair_b.pair_id,
                 pair_swap=pair_a.pair_id,
                 token_borrow=input_token_addr,
                 amount_borrow=raw_amount_in,
                 fee_borrow_bps=int(fee_b_dec * 10000),
                 fee_swap_bps=int(fee_a_dec * 10000),
                 min_profit=0,
                 dry_run=False,
                 rpc_url=manager.rpc_urls[0],
                 private_key=os.getenv("SKIM_PRIVATE_KEY", ""),
                 gas_price_gwei=args.gas_price_gwei,
                 monstrosity_address=MONSTROSITY_ADDRESS,
                 aave_pool_address=AAVE_V3_POOL_ADDRESS
             )

# --- Main ---

async def main_async():
    parser = argparse.ArgumentParser(description="Async Flash Arb Scanner")
    parser.add_argument("--db-path", default="skim_pairs.db")
    parser.add_argument("--dexes", default="uniswapv2,camelot,sushiswapv2,pancakeswap")
    parser.add_argument("--rpc-urls", default=os.getenv("ARBITRUM_RPC_URLS", ""))
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--loop", action="store_true")

    parser.add_argument("--triangular", action="store_true")
    parser.add_argument("--triangular-dump", default="")
    parser.add_argument("--triangular-auto-execute", action="store_true")
    parser.add_argument("--triangular-safety-bps", type=Decimal, default=Decimal("10"))
    parser.add_argument("--triangular-simulate-all", action="store_true")
    parser.add_argument("--triangular-allow-v3", action="store_true")
    parser.add_argument("--triangular-v3-amount-in", type=Decimal, default=Decimal("0.1"))

    parser.add_argument("--auto-execute", action="store_true")
    parser.add_argument("--auto-execute-allow-any", action="store_true")
    parser.add_argument("--min-profit", type=Decimal, default=Decimal("0"))
    parser.add_argument("--min-net-profit-usd", type=Decimal, default=Decimal("0.01"))
    parser.add_argument("--max-trade-frac", type=Decimal, default=Decimal("1.0"))
    parser.add_argument("--gas-price-gwei", type=Decimal, default=Decimal("0.02"))
    parser.add_argument("--gas-units", type=int, default=500000)

    parser.add_argument("--fee-uniswap", type=Decimal, default=Decimal("0.003"))
    parser.add_argument("--fee-camelot", type=Decimal, default=Decimal("0.005"))
    parser.add_argument("--fee-sushiswap", type=Decimal, default=Decimal("0.005"))
    parser.add_argument("--fee-pancakeswap", type=Decimal, default=Decimal("0.0025"))

    parser.add_argument("--ignore-addresses", default="")
    parser.add_argument("--allow-addresses", default="")
    parser.add_argument("--ignore-tokens", default="")
    parser.add_argument("--settle-token", default="none")
    parser.add_argument("--settle-fee-bps", type=int, default=30)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--watchlist", default="")
    parser.add_argument("--focus-top-reserve", type=int, default=0)
    parser.add_argument("--focus-top-volume", type=int, default=0)
    parser.add_argument("--monstrosity-addr", default=MONSTROSITY_ADDRESS)
    parser.add_argument("--aave-pool", default=AAVE_V3_POOL_ADDRESS)
    parser.add_argument("--min-pair-liquidity-usd", type=Decimal, default=Decimal("0"))

    args = parser.parse_args()

    dexes = [d.strip() for d in args.dexes.split(",")]

    logger.info("Loading pairs...")
    conn = sqlite3.connect(args.db_path)
    all_pairs = []

    if args.watchlist:
        all_pairs = load_pairs_from_watchlist(conn, args.watchlist)
    else:
        for dex in dexes:
            all_pairs.extend(load_pairs(conn, dex, args.max_pairs))

    # Simple ignore list filtering on load
    ignore_addresses = parse_ignore_addresses(args.ignore_addresses + "," + args.ignore_tokens)
    if ignore_addresses:
        all_pairs = [p for p in all_pairs if not (is_ignored_address(p.pair_id, ignore_addresses) or is_ignored_address(p.token0, ignore_addresses) or is_ignored_address(p.token1, ignore_addresses))]

    logger.info(f"Loaded {len(all_pairs)} pairs total.")

    # Build structures
    pair_index = build_pair_index(all_pairs)
    by_tokens = {}
    for p in all_pairs:
        key = tuple(sorted([p.token0.lower(), p.token1.lower()]))
        by_tokens.setdefault(key, {})[p.dex] = p

    rpc_urls = build_rpc_pool(args.rpc_urls)
    manager = AsyncRPCManager(rpc_urls)

    fee_by_dex = {
        "uniswapv2": args.fee_uniswap,
        "camelot": args.fee_camelot,
        "sushiswapv2": args.fee_sushiswap,
        "pancakeswap": args.fee_pancakeswap,
    }

    try:
        while True:
            logger.info("--- Start Scan Iteration ---")
            reserve_cache = {}

            # Narrow Scan Keys
            narrow_keys = [k for k, v in by_tokens.items() if sum(1 for dex in dexes if dex in v) >= 2]

            tasks = []
            if narrow_keys:
                tasks.append(narrow_scan_async(manager, narrow_keys, by_tokens, args, fee_by_dex))

            if args.triangular:
                tasks.append(triangular_scan_async(manager, all_pairs, pair_index, args.triangular_dump, args, fee_by_dex, reserve_cache))

            if tasks:
                await asyncio.gather(*tasks)

            if not args.loop:
                break
            await asyncio.sleep(0.5)

    finally:
        await manager.close()

# Alias for export
execute_triangular_trade = execute_triangular_trade_async
execute_path_trade = execute_path_trade_async

if __name__ == "__main__":
    asyncio.run(main_async())

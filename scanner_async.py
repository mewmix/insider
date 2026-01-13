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
from web3 import Web3
from eth_abi import encode as eth_abi_encode

from ignore_list import parse_ignore_addresses, is_ignored_address
from policy import parse_allow_addresses, is_allowed_address
from scanner_config import RPC_ENDPOINTS, normalize_rpc_url

load_dotenv()
getcontext().prec = 60

# --- Configuration ---

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

GET_RESERVES_SIG = "0x0902f1ac"

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

# --- RPC Manager ---

class AsyncRPCManager:
    def __init__(self, rpc_urls: List[str]):
        self.rpc_urls = rpc_urls
        self.index = 0
        self.lock = asyncio.Lock()
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
        
        raise RuntimeError(f"All RPCs failed. Last error: {last_error}")

    async def close(self):
        if self.session:
            await self.session.close()

# --- Async Helper Functions ---

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

# --- Core Logic (Ported) ---

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

    def profit_fn(val: Decimal) -> Decimal:
        out = triangle_out(val, start_token, token_b, token_c, pair_ab, pair_bc, pair_ca, fee_by_dex)
        return out - val

    lo = Decimal(0)
    hi = max_in
    for _ in range(30):
        m1 = lo + (hi - lo) / Decimal(3)
        m2 = hi - (hi - lo) / Decimal(3)
        p1 = profit_fn(m1)
        p2 = profit_fn(m2)
        if p1 > p2:
            hi = m2
        else:
            lo = m1
    
    best_in = (lo + hi) / Decimal(2)
    best_profit = profit_fn(best_in)
    return best_in, best_profit

# --- Main Scan Task ---

async def triangular_scan_async(
    manager: AsyncRPCManager,
    pairs: List[PairData],
    dump_path: str,
    args: argparse.Namespace,
    fee_by_dex: Dict[str, Decimal],
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
    sem = asyncio.Semaphore(100) 
    
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
                            await asyncio.gather(
                                fetch_pair_reserves(manager, pair_ab),
                                fetch_pair_reserves(manager, pair_bc),
                                fetch_pair_reserves(manager, pair_ca)
                            )
                            
                            amt_in, profit = best_triangle_arb(
                                start_token, b, c, pair_ab, pair_bc, pair_ca, fee_by_dex, args.max_trade_frac
                            )
                            
                            if profit <= 0: continue
                            
                            safety_mult = (Decimal(10000) - args.triangular_safety_bps) / Decimal(10000)
                            amt_out_safe = (amt_in + profit) * safety_mult
                            profit_safe = amt_out_safe - amt_in
                            
                            if profit_safe <= 0: continue
                            
                            logger.info(f"OPP: {start_token}->{b}->{c} | Profit: {profit_safe} {start_token}")
                            
                            if dump_path:
                                decimals = pair_ab.token0_decimals if pair_ab.token0.lower() == start_token.lower() else pair_ab.token1_decimals
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

                            if args.auto_execute or args.triangular_auto_execute:
                                if not args.auto_execute_allow_any and not allow_addresses:
                                    logger.warning("Auto-execute skipped: allowlist empty")
                                    continue
                                
                                # Prepare execution args
                                hop_types = tuple("v3" if "v3" in p.dex else "v2" for p in (pair_ab, pair_bc, pair_ca))
                                
                                decimals = pair_ab.token0_decimals if pair_ab.token0.lower() == start_token.lower() else pair_ab.token1_decimals
                                raw_amount_in = int(amt_in * (Decimal(10) ** decimals))
                                
                                # Call sync execution in thread
                                from scanner import execute_triangular_trade
                                
                                # Use first RPC for execution
                                exec_rpc = manager.rpc_urls[0]
                                
                                logger.info(f"Auto-executing triangular arb...")
                                await asyncio.to_thread(
                                    execute_triangular_trade,
                                    start_token, b, c, pair_ab, pair_bc, pair_ca,
                                    hop_types,
                                    raw_amount_in,
                                    exec_rpc,
                                    os.getenv("SKIM_PRIVATE_KEY", ""),
                                    Decimal("0.02"), # Gas Price fixed/dummy or arg
                                    False, # dry_run
                                    MONSTROSITY_ADDRESS,
                                    AAVE_V3_POOL_ADDRESS,
                                    fee_by_dex,
                                    args.triangular_safety_bps,
                                    0 # min_profit_weth
                                )

    for t in scan_tokens:
        tasks.append(asyncio.create_task(check_cycle(t)))
    
    await asyncio.gather(*tasks)


async def main_async():
    parser = argparse.ArgumentParser(description="Async Flash Arb Scanner")
    parser.add_argument("--db-path", default="skim_pairs.db")
    parser.add_argument("--triangular", action="store_true")
    parser.add_argument("--triangular-dump", default="")
    parser.add_argument("--dexes", default="uniswapv2,camelot,sushiswapv2,pancakeswap,uniswapv3")
    parser.add_argument("--rpc-urls", default=os.getenv("ARBITRUM_RPC_URLS", ""))
    parser.add_argument("--ignore-addresses", default="")
    parser.add_argument("--allow-addresses", default="")
    parser.add_argument("--max-trade-frac", type=Decimal, default=Decimal("1.0"))
    parser.add_argument("--triangular-safety-bps", type=Decimal, default=Decimal("10"))
    parser.add_argument("--auto-execute", action="store_true")
    parser.add_argument("--min-net-profit-usd", type=Decimal, default=Decimal("0.01"))
    parser.add_argument("--max-pairs", type=int, default=0, help="Limit pairs per dex (0 = all).")
    parser.add_argument("--triangular-auto-execute", action="store_true", help="Execute triangular routes.")
    parser.add_argument("--auto-execute-allow-any", action="store_true", help="Allow auto-execute without allowlist.")
    
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    dexes = [d.strip() for d in args.dexes.split(",")]
    
    # Simple direct load from DB instead of importing synchronous load_pairs
    # This avoids circular imports or syncing issues
    all_pairs = []
    for dex in dexes:
        query = "SELECT pair_id, token0, token1, token0_symbol, token1_symbol, token0_decimals, token1_decimals FROM pairs WHERE dex = ?"
        params = [dex]
        if args.max_pairs > 0:
            query += " LIMIT ?"
            params.append(args.max_pairs)
            
        rows = conn.execute(query, params).fetchall()
        for row in rows:
            all_pairs.append(PairData(
                dex=dex,
                pair_id=row[0],
                token0=row[1],
                token1=row[2],
                token0_symbol=row[3] or "",
                token1_symbol=row[4] or "",
                token0_decimals=int(row[5] or 18),
                token1_decimals=int(row[6] or 18),
                reserve0=Decimal(0),
                reserve1=Decimal(0),
            ))
    
    logger.info(f"Loaded {len(all_pairs)} pairs.")

    from scanner import build_rpc_pool
    rpc_urls = build_rpc_pool(args.rpc_urls)
    manager = AsyncRPCManager(rpc_urls)

    fee_by_dex = {
        "uniswapv2": Decimal("0.003"),
        "camelot": Decimal("0.005"),
        "sushiswapv2": Decimal("0.005"),
        "pancakeswap": Decimal("0.0025"),
        "uniswapv3": Decimal("0.003"),
    }

    try:
        # Loop functionality implied by request "run this... can be async"
        # We will loop if arg is passed? Or just once?
        # User cmd: `python3 -u scanner.py ... --loop` (implicit in original usage, but not in provided arg list here)
        # I'll run once.
        if args.triangular:
            await triangular_scan_async(manager, all_pairs, args.triangular_dump, args, fee_by_dex)
        else:
            logger.warning("No scan mode selected (use --triangular)")
    finally:
        await manager.close()

if __name__ == "__main__":
    asyncio.run(main_async())

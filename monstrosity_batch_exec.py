import argparse
import json
import os
import sys
from decimal import Decimal
from typing import Dict, Iterator

from dotenv import load_dotenv

from ignore_list import parse_ignore_addresses, is_ignored_address
from monstrosity_scanner import PairData, execute_triangular_trade
from policy import parse_allow_addresses, is_allowed_address


load_dotenv()


def read_jsonl(path: str) -> Iterator[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_pair(pair_id: str, token0: str, token1: str, dex: str, d0: int, d1: int) -> PairData:
    return PairData(
        dex=dex,
        pair_id=pair_id,
        token0=token0,
        token1=token1,
        token0_symbol="",
        token1_symbol="",
        token0_decimals=d0,
        token1_decimals=d1,
        reserve0=Decimal(0),
        reserve1=Decimal(0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch simulate/execute Monstrosity triangular opps.")
    parser.add_argument("--opps-file", required=True, help="JSONL file with opportunities.")
    parser.add_argument("--rpc-url", default=os.getenv("ARBITRUM_RPC_URL", "https://arb1.arbitrum.io/rpc"))
    parser.add_argument("--monstrosity-addr", required=True, help="Monstrosity contract address.")
    parser.add_argument("--aave-pool", required=True, help="Aave V3 pool address.")
    parser.add_argument("--gas-price-gwei", type=Decimal, default=Decimal("0.02"))
    parser.add_argument("--max", type=int, default=50, help="Max opps to process.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate only.")
    parser.add_argument("--auto-execute", action="store_true", help="Execute after successful sim.")
    parser.add_argument("--ignore-addresses", default="", help="Comma-separated addresses to ignore.")
    parser.add_argument("--allow-addresses", default="", help="Comma-separated addresses to allow.")
    parser.add_argument("--auto-execute-allow-any", action="store_true", help="Allow auto-execute without allowlist.")
    args = parser.parse_args()

    private_key = os.getenv("SKIM_PRIVATE_KEY", "")
    if not private_key:
        print("SKIM_PRIVATE_KEY required", file=sys.stderr)
        sys.exit(1)

    ignore_addresses = parse_ignore_addresses(args.ignore_addresses)
    allow_addresses = parse_allow_addresses(args.allow_addresses)

    processed = 0
    for opp in read_jsonl(args.opps_file):
        if processed >= args.max:
            break
        start_token = str(opp["start_token"]).lower()
        token_b = str(opp["token_b"]).lower()
        token_c = str(opp["token_c"]).lower()
        pair_ab = str(opp["pair_ab"])
        pair_bc = str(opp["pair_bc"])
        pair_ca = str(opp["pair_ca"])
        amount_in_raw = int(opp["amount_in_raw"])

        if any(
            is_ignored_address(addr, ignore_addresses)
            for addr in (start_token, token_b, token_c, pair_ab, pair_bc, pair_ca)
        ):
            continue

        if args.auto_execute and not args.auto_execute_allow_any and not allow_addresses:
            print("auto-execute blocked: allowlist empty", file=sys.stderr)
            sys.exit(2)

        if args.auto_execute and allow_addresses:
            if not all(
                is_allowed_address(addr, allow_addresses, allow_any=False)
                for addr in (start_token, token_b, token_c, pair_ab, pair_bc, pair_ca)
            ):
                continue

        dex_ab = str(opp.get("dex_ab", ""))
        dex_bc = str(opp.get("dex_bc", ""))
        dex_ca = str(opp.get("dex_ca", ""))

        d_start = int(opp.get("start_token_decimals", 18))
        d_b = int(opp.get("token_b_decimals", 18))
        d_c = int(opp.get("token_c_decimals", 18))

        pair_ab_obj = build_pair(pair_ab, start_token, token_b, dex_ab, d_start, d_b)
        pair_bc_obj = build_pair(pair_bc, token_b, token_c, dex_bc, d_b, d_c)
        pair_ca_obj = build_pair(pair_ca, token_c, start_token, dex_ca, d_c, d_start)

        fee_by_dex = {
            dex_ab: Decimal(int(opp.get("fee_bps_ab", 30))) / Decimal(10000),
            dex_bc: Decimal(int(opp.get("fee_bps_bc", 30))) / Decimal(10000),
            dex_ca: Decimal(int(opp.get("fee_bps_ca", 30))) / Decimal(10000),
        }
        safety_bps = Decimal(int(opp.get("safety_bps", 10)))
        min_profit_weth = int(opp.get("min_profit_weth_raw", 0))

        sim_ok = execute_triangular_trade(
            start_token=start_token,
            token_b=token_b,
            token_c=token_c,
            pair_ab=pair_ab_obj,
            pair_bc=pair_bc_obj,
            pair_ca=pair_ca_obj,
            amount_in=amount_in_raw,
            rpc_url=args.rpc_url,
            private_key=private_key,
            gas_price_gwei=args.gas_price_gwei,
            dry_run=True,
            monstrosity_address=args.monstrosity_addr,
            aave_pool_address=args.aave_pool,
            fee_by_dex=fee_by_dex,
            safety_bps=safety_bps,
            min_profit_weth=min_profit_weth,
        )
        print(f"sim={sim_ok} start={start_token} b={token_b} c={token_c} in={amount_in_raw}")
        processed += 1
        if sim_ok and args.auto_execute and not args.dry_run:
            exec_ok = execute_triangular_trade(
                start_token=start_token,
                token_b=token_b,
                token_c=token_c,
                pair_ab=pair_ab_obj,
                pair_bc=pair_bc_obj,
                pair_ca=pair_ca_obj,
                amount_in=amount_in_raw,
                rpc_url=args.rpc_url,
                private_key=private_key,
                gas_price_gwei=args.gas_price_gwei,
                dry_run=False,
                monstrosity_address=args.monstrosity_addr,
                aave_pool_address=args.aave_pool,
                fee_by_dex=fee_by_dex,
                safety_bps=safety_bps,
                min_profit_weth=min_profit_weth,
            )
            print(f"execute={exec_ok} start={start_token} b={token_b} c={token_c}")


if __name__ == "__main__":
    main()

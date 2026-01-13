import argparse
import os
import sys
from typing import List

from dotenv import load_dotenv
from web3 import Web3

from skim_scanner import RPC_ENDPOINTS, normalize_rpc_url
from ignore_list import parse_ignore_addresses, is_ignored_address
from policy import parse_allow_addresses, is_allowed_address


load_dotenv()

PAIR_ABI = [
    {
        "inputs": [{"internalType": "address", "name": "to", "type": "address"}],
        "name": "skim",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]


def build_rpc_pool(rpc_urls: str) -> List[str]:
    if rpc_urls:
        urls = [normalize_rpc_url(url.strip()) for url in rpc_urls.split(",") if url.strip()]
    else:
        env_url = normalize_rpc_url(os.getenv("ARBITRUM_RPC_URL", "https://arb1.arbitrum.io/rpc"))
        urls = []
        if env_url:
            urls.append(env_url)
        urls.extend(
            normalize_rpc_url(url)
            for url in RPC_ENDPOINTS.values()
            if url.startswith("http")
        )
    deduped: List[str] = []
    seen = set()
    for url in urls:
        if not url or url in seen:
            continue
        deduped.append(url)
        seen.add(url)
    return deduped


def send_skim(rpc_urls: List[str], pair: str, to_addr: str, private_key: str) -> None:
    last_err = None
    for rpc_url in rpc_urls:
        try:
            w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 15}))
            if not w3.is_connected():
                raise RuntimeError("RPC not connected")
            account = w3.eth.account.from_key(private_key)
            # Force a basic chain check to fail early on bad endpoints.
            _ = w3.eth.chain_id
            contract = w3.eth.contract(address=Web3.to_checksum_address(pair), abi=PAIR_ABI)
            nonce = w3.eth.get_transaction_count(account.address)
            latest_block = w3.eth.get_block("latest")
            base_fee = latest_block.get("baseFeePerGas")
            tx_params = {
                "from": account.address,
                "nonce": nonce,
                "chainId": 42161,
            }
            if base_fee is not None:
                # Ensure max fee clears the current base fee for EIP-1559 networks.
                try:
                    priority_fee = w3.eth.max_priority_fee
                except Exception:
                    priority_fee = w3.to_wei(0.01, "gwei")
                tx_params.update(
                    {
                        "type": 2,
                        "maxPriorityFeePerGas": int(priority_fee),
                        "maxFeePerGas": int(base_fee * 2 + priority_fee),
                    }
                )
            else:
                tx_params["gasPrice"] = w3.eth.gas_price
            tx = contract.functions.skim(Web3.to_checksum_address(to_addr)).build_transaction(
                tx_params
            )
            gas = w3.eth.estimate_gas(tx)
            tx["gas"] = int(gas * 1.2)
            signed = w3.eth.account.sign_transaction(tx, private_key)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            print(f"rpc={rpc_url}")
            print(f"tx_hash={tx_hash.hex()}")
            return
        except Exception as exc:
            print(f"rpc_failed={rpc_url} err={exc}", file=sys.stderr)
            last_err = exc
            continue
    raise RuntimeError(f"failed to send skim tx: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute skim on a v2 pair.")
    parser.add_argument("--pair", required=True, help="Pair address to skim.")
    parser.add_argument(
        "--to",
        default="",
        help="Recipient address (defaults to private key address).",
    )
    parser.add_argument(
        "--rpc-urls",
        default=os.getenv("ARBITRUM_RPC_URLS", ""),
        help="Comma-separated RPC URLs to rotate across.",
    )
    parser.add_argument("--ignore-addresses", default="", help="Comma-separated addresses to ignore.")
    parser.add_argument("--allow-addresses", default="", help="Comma-separated addresses to allow.")
    args = parser.parse_args()

    private_key = os.getenv("SKIM_PRIVATE_KEY")
    if not private_key:
        print("SKIM_PRIVATE_KEY is required in .env", file=sys.stderr)
        sys.exit(1)

    w3_tmp = Web3()
    from_addr = w3_tmp.eth.account.from_key(private_key).address
    to_addr = args.to or from_addr
    ignore_addresses = parse_ignore_addresses(args.ignore_addresses)
    allow_addresses = parse_allow_addresses(args.allow_addresses)
    if is_ignored_address(args.pair, ignore_addresses):
        print("execution blocked: pair is in ignore list", file=sys.stderr)
        sys.exit(2)
    if allow_addresses and not is_allowed_address(args.pair, allow_addresses, allow_any=False):
        print("execution blocked: pair not in allow list", file=sys.stderr)
        sys.exit(2)

    rpc_urls = build_rpc_pool(args.rpc_urls)
    send_skim(rpc_urls, args.pair, to_addr, private_key)


if __name__ == "__main__":
    main()

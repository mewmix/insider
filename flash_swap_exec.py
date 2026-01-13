import argparse
import os
import sys
import json
import time
from dotenv import load_dotenv
from web3 import Web3
from ignore_list import parse_ignore_addresses, is_ignored_address
from policy import parse_allow_addresses, is_allowed_address

load_dotenv()

# FlashArb contract ABI (only the execute function is needed for calling)
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

def main() -> None:
    parser = argparse.ArgumentParser(description="Flash swap executor.")
    parser.add_argument("--pair-borrow", required=True, help="Pair to borrow from.")
    parser.add_argument("--pair-swap", required=True, help="Pair to swap on.")
    parser.add_argument("--token-borrow", required=True, help="Token to borrow.")
    parser.add_argument("--amount-borrow", required=True, type=int, help="Borrow amount (raw units).")
    parser.add_argument("--fee-borrow-bps", default=30, type=int, help="Borrow pair fee bps.")
    parser.add_argument("--fee-swap-bps", default=30, type=int, help="Swap pair fee bps.")
    parser.add_argument("--min-profit", default=0, type=int, help="Minimum profit (raw units).")
    parser.add_argument("--contract-address", required=True, help="Deployed FlashArb contract address.")
    parser.add_argument("--rpc-url", default=os.getenv("ARBITRUM_RPC_URL", "https://arb1.arbitrum.io/rpc"))
    parser.add_argument("--private-key", default=os.getenv("SKIM_PRIVATE_KEY", ""), help="Private key for execution.")
    parser.add_argument("--chain-id", default=42161, type=int, help="Chain ID (default: 42161 for Arbitrum).")
    parser.add_argument("--dry-run", action="store_true", help="Simulate only (eth_call/estimateGas).")
    parser.add_argument("--ignore-addresses", default="", help="Comma-separated addresses to ignore.")
    parser.add_argument("--allow-addresses", default="", help="Comma-separated addresses to allow.")
    args = parser.parse_args()

    if not args.private_key:
        print("SKIM_PRIVATE_KEY required", file=sys.stderr)
        sys.exit(1)

    ignore_addresses = parse_ignore_addresses(args.ignore_addresses)
    allow_addresses = parse_allow_addresses(args.allow_addresses)
    if any(
        is_ignored_address(addr, ignore_addresses)
        for addr in (args.pair_borrow, args.pair_swap, args.token_borrow)
    ):
        print("execution blocked: address is in ignore list", file=sys.stderr)
        sys.exit(2)
    if allow_addresses and not all(
        is_allowed_address(addr, allow_addresses, allow_any=False)
        for addr in (args.pair_borrow, args.pair_swap, args.token_borrow)
    ):
        print("execution blocked: address not in allow list", file=sys.stderr)
        sys.exit(2)

    w3 = Web3(Web3.HTTPProvider(args.rpc_url, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        print("RPC not connected", file=sys.stderr)
        sys.exit(1)

    account = w3.eth.account.from_key(args.private_key)
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(args.contract_address),
        abi=FLASH_ARB_ABI
    )

    pair_borrow = Web3.to_checksum_address(args.pair_borrow)
    pair_swap = Web3.to_checksum_address(args.pair_swap)
    token_borrow = Web3.to_checksum_address(args.token_borrow)

    print(f"Plan: Borrow {args.amount_borrow} of {token_borrow} from {pair_borrow}, swap on {pair_swap}")

    # Build transaction
    tx_func = contract.functions.execute(
        pair_borrow,
        pair_swap,
        token_borrow,
        args.amount_borrow,
        args.fee_borrow_bps,
        args.fee_swap_bps,
        args.min_profit
    )

    # Estimate gas
    try:
        gas_estimate = tx_func.estimate_gas({"from": account.address})
        print(f"Gas estimate: {gas_estimate}")
    except Exception as e:
        print(f"Gas estimation failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("Dry run complete.")
        return

    # Send transaction
    nonce = w3.eth.get_transaction_count(account.address)

    # Priority fee logic for Arbitrum (EIP-1559)
    latest_block = w3.eth.get_block("latest")
    base_fee = latest_block.get("baseFeePerGas")

    tx_params = {
        "from": account.address,
        "nonce": nonce,
        "chainId": args.chain_id,
        "gas": int(gas_estimate * 1.2), # Add 20% buffer
    }

    if base_fee is not None:
         # Arbitrum usually has low priority fee requirements, but we want to be safe
        max_priority_fee = w3.to_wei(0.1, "gwei")
        max_fee = int(base_fee * 1.35 + max_priority_fee)
        tx_params["maxPriorityFeePerGas"] = max_priority_fee
        tx_params["maxFeePerGas"] = max_fee
        tx_params["type"] = 2
    else:
        tx_params["gasPrice"] = int(w3.eth.gas_price * 1.1)

    signed_tx = w3.eth.account.sign_transaction(tx_func.build_transaction(tx_params), args.private_key)
    print("Sending transaction...")
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print(f"Transaction sent: {tx_hash.hex()}")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    if receipt.status == 1:
        print("Transaction successful!")
    else:
        print("Transaction failed/reverted.")
        sys.exit(1)

if __name__ == "__main__":
    main()

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from web3 import Web3
from solcx import compile_standard, install_solc


load_dotenv()

SOLC_VERSION = "0.8.20"


def compile_contract() -> Tuple[dict, str]:
    install_solc(SOLC_VERSION)
    source = Path("contracts/FlashArb.sol").read_text()
    compiled = compile_standard(
        {
            "language": "Solidity",
            "sources": {"FlashArb.sol": {"content": source}},
            "settings": {
                "optimizer": {"enabled": True, "runs": 200},
                "viaIR": True,
                "outputSelection": {"*": {"*": ["abi", "evm.bytecode"]}},
            },
        },
        solc_version=SOLC_VERSION,
    )
    contract = compiled["contracts"]["FlashArb.sol"]["FlashArb"]
    return contract["abi"], contract["evm"]["bytecode"]["object"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Flash swap gas simulation (deploy + estimate).")
    parser.add_argument("--pair-borrow", default="", help="Pair to borrow from.")
    parser.add_argument("--pair-swap", default="", help="Pair to swap on.")
    parser.add_argument("--token-borrow", default="", help="Token to borrow.")
    parser.add_argument("--amount-borrow", default=0, type=int, help="Borrow amount (raw units).")
    parser.add_argument("--fee-borrow-bps", default=30, type=int, help="Borrow pair fee bps.")
    parser.add_argument("--fee-swap-bps", default=30, type=int, help="Swap pair fee bps.")
    parser.add_argument("--min-profit", default=0, type=int, help="Minimum profit (raw units).")
    parser.add_argument("--rpc-url", default=os.getenv("ARBITRUM_RPC_URL", "https://arb1.arbitrum.io/rpc"))
    parser.add_argument("--deploy", action="store_true", help="Deploy contract before estimate.")
    parser.add_argument("--contract-address", default="", help="Use an existing deployed contract.")
    parser.add_argument("--batch-file", default="", help="JSON file with batch scenarios.")
    parser.add_argument(
        "--deploy-gas",
        type=int,
        default=0,
        help="Explicit gas limit for deploy (skips estimate if set).",
    )
    parser.add_argument("--private-key", default=os.getenv("SKIM_PRIVATE_KEY", ""), help="Private key for deploy.")
    args = parser.parse_args()

    w3 = Web3(Web3.HTTPProvider(args.rpc_url, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        print("rpc not connected", file=sys.stderr)
        sys.exit(1)

    abi, bytecode = compile_contract()
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)

    scenarios = []
    if args.batch_file:
        import json

        with open(args.batch_file, "r") as handle:
            scenarios = json.load(handle)
    else:
        if not (args.pair_borrow and args.pair_swap and args.token_borrow and args.amount_borrow):
            print("pair-borrow, pair-swap, token-borrow, amount-borrow are required without --batch-file", file=sys.stderr)
            sys.exit(2)
        scenarios = [
            {
                "pair_borrow": args.pair_borrow,
                "pair_swap": args.pair_swap,
                "token_borrow": args.token_borrow,
                "amount_borrow": args.amount_borrow,
                "fee_borrow_bps": args.fee_borrow_bps,
                "fee_swap_bps": args.fee_swap_bps,
                "min_profit": args.min_profit,
            }
        ]

    if not args.contract_address and not args.deploy:
        print("deploy flag not set; skipping deployment", file=sys.stderr)
        print("deploy_gas_estimate", contract.constructor().estimate_gas())
        return

    if not args.private_key:
        print("SKIM_PRIVATE_KEY required for deploy", file=sys.stderr)
        sys.exit(1)

    account = w3.eth.account.from_key(args.private_key)
    contract_address = args.contract_address
    if not contract_address:
        nonce = w3.eth.get_transaction_count(account.address)
        gas_price = int(w3.eth.gas_price * 1.5)
        tx_params = {
            "from": account.address,
            "nonce": nonce,
            "gasPrice": gas_price,
            "chainId": 42161,
        }
        if args.deploy_gas:
            tx_params["gas"] = args.deploy_gas
            tx = contract.constructor().build_transaction(tx_params)
        else:
            tx = contract.constructor().build_transaction(tx_params)
            try:
                tx["gas"] = int(w3.eth.estimate_gas(tx) * 1.2)
            except Exception:
                tx["gas"] = 2_500_000
        signed = w3.eth.account.sign_transaction(tx, args.private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        contract_address = receipt.contractAddress
        print("deploy_tx", tx_hash.hex())
        print("contract", contract_address)

    deployed = w3.eth.contract(
        address=Web3.to_checksum_address(contract_address),
        abi=abi,
    )
    for scenario in scenarios:
        call = deployed.functions.execute(
            Web3.to_checksum_address(scenario["pair_borrow"]),
            Web3.to_checksum_address(scenario["pair_swap"]),
            Web3.to_checksum_address(scenario["token_borrow"]),
            int(scenario["amount_borrow"]),
            int(scenario.get("fee_borrow_bps", args.fee_borrow_bps)),
            int(scenario.get("fee_swap_bps", args.fee_swap_bps)),
            int(scenario.get("min_profit", args.min_profit)),
        )
        gas_estimate = call.estimate_gas({"from": account.address})
        print(
            "execute_gas_estimate",
            scenario["pair_borrow"],
            scenario["pair_swap"],
            gas_estimate,
        )


if __name__ == "__main__":
    main()

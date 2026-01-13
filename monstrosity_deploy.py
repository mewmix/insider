import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

from dotenv import load_dotenv
from solcx import compile_standard, install_solc
from web3 import Web3


load_dotenv()

SOLC_VERSION = "0.8.20"
DEFAULT_RPC_URL = os.getenv("ANVIL_RPC_URL", "http://127.0.0.1:8545")
DEFAULT_WETH = os.getenv(
    "WETH_ADDRESS",
    "0x82af49447d8a07e3bd95bd0d56f35241523fbab1",
)


def load_sources() -> Dict[str, Dict[str, str]]:
    sources: Dict[str, Dict[str, str]] = {}
    contracts_root = Path("contracts")

    for path in [contracts_root / "Monstrosity.sol"]:
        sources[str(path)] = {"content": path.read_text()}

    for path in (contracts_root / "interfaces").glob("*.sol"):
        sources[str(path)] = {"content": path.read_text()}

    return sources


def compile_monstrosity() -> Tuple[dict, str]:
    install_solc(SOLC_VERSION)
    compiled = compile_standard(
        {
            "language": "Solidity",
            "sources": load_sources(),
            "settings": {
                "optimizer": {"enabled": True, "runs": 200},
                "outputSelection": {"*": {"*": ["abi", "evm.bytecode"]}},
            },
        },
        solc_version=SOLC_VERSION,
    )
    contract = compiled["contracts"]["contracts/Monstrosity.sol"]["Monstrosity"]
    return contract["abi"], contract["evm"]["bytecode"]["object"]


def update_json(path: Path, updates: Dict[str, object]) -> None:
    data: Dict[str, object] = {}
    if path.exists():
        data = json.loads(path.read_text())
    data.update(updates)
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy Monstrosity on an Anvil fork.")
    parser.add_argument("--rpc-url", default=DEFAULT_RPC_URL, help="RPC URL (anvil fork).")
    parser.add_argument("--private-key", default=os.getenv("ANVIL_PRIVATE_KEY", os.getenv("SKIM_PRIVATE_KEY", "")))
    parser.add_argument("--weth-address", default=DEFAULT_WETH, help="WETH address for constructor.")
    parser.add_argument("--json-file", default="arb_contracts.json", help="JSON file to store address.")
    parser.add_argument("--no-write-json", action="store_true", help="Skip writing address to JSON.")
    parser.add_argument("--gas-multiplier", type=float, default=1.2, help="Multiplier for gas estimate.")
    args = parser.parse_args()

    w3 = Web3(Web3.HTTPProvider(args.rpc_url, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        print(f"rpc not connected: {args.rpc_url}", file=sys.stderr)
        sys.exit(1)

    if not args.private_key:
        print("ANVIL_PRIVATE_KEY or SKIM_PRIVATE_KEY required for deploy", file=sys.stderr)
        sys.exit(1)

    abi, bytecode = compile_monstrosity()
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)

    account = w3.eth.account.from_key(args.private_key)
    chain_id = w3.eth.chain_id

    tx = contract.constructor(Web3.to_checksum_address(args.weth_address)).build_transaction(
        {
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address),
            "gasPrice": int(w3.eth.gas_price),
            "chainId": chain_id,
        }
    )
    try:
        tx["gas"] = int(w3.eth.estimate_gas(tx) * args.gas_multiplier)
    except Exception:
        tx["gas"] = 3_000_000

    signed = w3.eth.account.sign_transaction(tx, args.private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    contract_address = receipt.contractAddress

    print("deploy_tx", tx_hash.hex())
    print("monstrosity", contract_address)

    if not args.no_write_json:
        update_json(
            Path(args.json_file),
            {
                "monstrosity": contract_address,
                "monstrosity_chain_id": chain_id,
            },
        )


if __name__ == "__main__":
    main()

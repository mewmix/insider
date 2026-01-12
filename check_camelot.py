import os
from web3 import Web3

RPC_URL = os.getenv('ARBITRUM_RPC_URL','https://arb1.arbitrum.io/rpc')
w3 = Web3(Web3.HTTPProvider(RPC_URL))

pair_addr = '0x19d51dc52e52407656b40b197b1bbe14294b955b'

# Try different getReserves signatures
signatures = [
    "getReserves()", # UniV2 standard: (uint112, uint112, uint32)
]

# Camelot V2 often returns 4 values: (uint112, uint112, uint112, uint112)
# Or it has separate fee functions.

pair = w3.eth.contract(address=Web3.to_checksum_address(pair_addr), abi=[
    {"constant": True, "inputs": [], "name": "getReserves", "outputs": [{"name": "", "type": "uint112"}, {"name": "", "type": "uint112"}, {"name": "", "type": "uint112"}, {"name": "", "type": "uint112"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "token0", "outputs": [{"name": "", "type": "address"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "token1", "outputs": [{"name": "", "type": "address"}], "type": "function"}
])

try:
    res = pair.functions.getReserves().call()
    print(f"Reserves: {res}")
except Exception as e:
    print(f"Failed getReserves (4 values): {e}")
    # Try 3 values
    pair3 = w3.eth.contract(address=Web3.to_checksum_address(pair_addr), abi=[
        {"constant": True, "inputs": [], "name": "getReserves", "outputs": [{"name": "", "type": "uint112"}, {"name": "", "type": "uint112"}, {"name": "", "type": "uint32"}], "type": "function"}
    ])
    try:
        res = pair3.functions.getReserves().call()
        print(f"Reserves (3 values): {res}")
    except Exception as e2:
        print(f"Failed getReserves (3 values): {e2}")

# Check for fee functions
try:
    fee = w3.eth.call({'to': Web3.to_checksum_address(pair_addr), 'data': w3.keccak(text="token0FeePercent()").hex()[:10]})
    print(f"token0FeePercent: {int(fee.hex(), 16)}")
except:
    pass

try:
    fee = w3.eth.call({'to': Web3.to_checksum_address(pair_addr), 'data': w3.keccak(text="token1FeePercent()").hex()[:10]})
    print(f"token1FeePercent: {int(fee.hex(), 16)}")
except:
    pass

import os
from web3 import Web3

RPC_URL = os.getenv('ARBITRUM_RPC_URL','https://arb1.arbitrum.io/rpc')
w3 = Web3(Web3.HTTPProvider(RPC_URL))

borrow_pair = '0x90bfecaef10aeedcd26c57a5232e9fccd5d8ce1b'
swap_pair = '0x19d51dc52e52407656b40b197b1bbe14294b955b'

def get_reserves(addr):
    # Try 3 values (UniV2)
    p3 = w3.eth.contract(address=Web3.to_checksum_address(addr), abi=[
        {"constant": True, "inputs": [], "name": "getReserves", "outputs": [{"name": "", "type": "uint112"}, {"name": "", "type": "uint112"}, {"name": "", "type": "uint32"}], "type": "function"}
    ])
    try:
        return p3.functions.getReserves().call()
    except:
        # Try 4 values (Camelot)
        p4 = w3.eth.contract(address=Web3.to_checksum_address(addr), abi=[
            {"constant": True, "inputs": [], "name": "getReserves", "outputs": [{"name": "", "type": "uint112"}, {"name": "", "type": "uint112"}, {"name": "", "type": "uint112"}, {"name": "", "type": "uint112"}], "type": "function"}
        ])
        return p4.functions.getReserves().call()

r_borrow = get_reserves(borrow_pair)
r_swap = get_reserves(swap_pair)

print(f"Borrow Pair {borrow_pair}: {r_borrow}")
print(f"Swap Pair {swap_pair}: {r_swap}")

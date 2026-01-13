import os
import httpx
from web3 import Web3

RPC_URL = os.getenv('ARBITRUM_RPC_URL','https://arb1.arbitrum.io/rpc')
w3 = Web3(Web3.HTTPProvider(RPC_URL))

pairs = [
    '0x90bfecaef10aeedcd26c57a5232e9fccd5d8ce1b',
    '0x19d51dc52e52407656b40b197b1bbe14294b955b'
]

erc20_abi = [
    {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"}
]

pair_abi = [
    {"constant": True, "inputs": [], "name": "token0", "outputs": [{"name": "", "type": "address"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "token1", "outputs": [{"name": "", "type": "address"}], "type": "function"}
]

for p_addr in pairs:
    p = w3.eth.contract(address=Web3.to_checksum_address(p_addr), abi=pair_abi)
    t0_addr = p.functions.token0().call()
    t1_addr = p.functions.token1().call()
    
    t0 = w3.eth.contract(address=t0_addr, abi=erc20_abi)
    t1 = w3.eth.contract(address=t1_addr, abi=erc20_abi)
    
    print(f"Pair: {p_addr}")
    try:
        print(f"  Token0: {t0_addr} ({t0.functions.symbol().call()})")
        print(f"  Token1: {t1_addr} ({t1.functions.symbol().call()})")
    except:
        print(f"  Token0: {t0_addr}")
        print(f"  Token1: {t1_addr}")

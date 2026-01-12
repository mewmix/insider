import os
from decimal import Decimal
from web3 import Web3

RPC_URL = os.getenv('ARBITRUM_RPC_URL','https://arb1.arbitrum.io/rpc')
w3 = Web3(Web3.HTTPProvider(RPC_URL))

# Configuration
PAIR_BORROW = '0x19d51dc52e52407656b40b197b1bbe14294b955b' # Camelot
PAIR_SWAP = '0x90bfecaef10aeedcd26c57a5232e9fccd5d8ce1b'   # UniV2
TOKEN_BORROW = '0x939727d85d99d0ac339bf1b76dfe30ca27c19067' # SIZE
AMOUNT_BORROW = 369859232953160699920
FEE_BORROW_BPS = 50 # Camelot
FEE_SWAP_BPS = 30   # UniV2
GAS_ESTIMATE = 204272

def get_reserves(addr):
    # Try Camelot (4 values) first as Pair Borrow is Camelot
    try:
        p = w3.eth.contract(address=Web3.to_checksum_address(addr), abi=[
            {"constant": True, "inputs": [], "name": "getReserves", "outputs": [{"name": "", "type": "uint112"}, {"name": "", "type": "uint112"}, {"name": "", "type": "uint112"}, {"name": "", "type": "uint112"}], "type": "function"}
        ])
        return p.functions.getReserves().call()[0:2] # Return first 2
    except:
        # Fallback to standard UniV2
        p = w3.eth.contract(address=Web3.to_checksum_address(addr), abi=[
            {"constant": True, "inputs": [], "name": "getReserves", "outputs": [{"name": "", "type": "uint112"}, {"name": "", "type": "uint112"}, {"name": "", "type": "uint32"}], "type": "function"}
        ])
        return p.functions.getReserves().call()[0:2]

def get_token_order(pair_addr, token_addr):
    p = w3.eth.contract(address=Web3.to_checksum_address(pair_addr), abi=[
        {"constant": True, "inputs": [], "name": "token0", "outputs": [{"name": "", "type": "address"}], "type": "function"}
    ])
    token0 = p.functions.token0().call()
    return 0 if token0.lower() == token_addr.lower() else 1

def get_amount_out(amount_in, reserve_in, reserve_out, fee_bps):
    amount_in_with_fee = amount_in * (10000 - fee_bps)
    numerator = amount_in_with_fee * reserve_out
    denominator = (reserve_in * 10000) + amount_in_with_fee
    return numerator // denominator

def get_amount_in(amount_out, reserve_in, reserve_out, fee_bps):
    numerator = reserve_in * amount_out * 10000
    denominator = (reserve_out - amount_out) * (10000 - fee_bps)
    return (numerator // denominator) + 1

# Fetch data
r_borrow = get_reserves(PAIR_BORROW)
r_swap = get_reserves(PAIR_SWAP)
borrow_token_idx = get_token_order(PAIR_BORROW, TOKEN_BORROW)

# Pair Borrow Reserves
# If tokenBorrow is token0, we borrow reserve0. Pay token is token1.
rb_in = r_borrow[1] if borrow_token_idx == 0 else r_borrow[0] # Reserve of PAY token (we pay this in)
rb_out = r_borrow[0] if borrow_token_idx == 0 else r_borrow[1] # Reserve of BORROW token (we borrow this out)

# Swap logic: We have Borrowed Token. We swap it on Pair Swap for Pay Token.
# Pair Swap Reserves
swap_token_idx = get_token_order(PAIR_SWAP, TOKEN_BORROW)
rs_in = r_swap[0] if swap_token_idx == 0 else r_swap[1] # Reserve of BORROW token (we put this in)
rs_out = r_swap[1] if swap_token_idx == 0 else r_swap[0] # Reserve of PAY token (we get this out)

print(f"Reserves Borrow Pair: Pay={rb_in}, Borrow={rb_out}")
print(f"Reserves Swap Pair: In={rs_in}, Out={rs_out}")

# 1. Calculate amount received from Swap Pair
amount_received = get_amount_out(AMOUNT_BORROW, rs_in, rs_out, FEE_SWAP_BPS)
print(f"Amount Borrowed (SIZE): {AMOUNT_BORROW}")
print(f"Amount Received (WETH): {amount_received}")

# 2. Calculate amount required to repay Borrow Pair
amount_required = get_amount_in(AMOUNT_BORROW, rb_in, rb_out, FEE_BORROW_BPS)
print(f"Amount Required (WETH): {amount_required}")

# 3. Profit
profit_raw = amount_received - amount_required
print(f"Gross Profit (Wei): {profit_raw}")

if profit_raw > 0:
    gas_price = w3.eth.gas_price
    tx_cost = GAS_ESTIMATE * gas_price
    net_profit = profit_raw - tx_cost
    print(f"Gas Price: {gas_price}")
    print(f"Tx Cost (Wei): {tx_cost}")
    print(f"Net Profit (Wei): {net_profit}")
    print(f"Net Profit (ETH): {Decimal(net_profit) / Decimal(10**18)}")
else:
    print("Trade is not profitable.")

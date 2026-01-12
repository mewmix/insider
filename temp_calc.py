import os
from decimal import Decimal
import httpx

RPC_URL = os.getenv('ARBITRUM_RPC_URL','https://arb1.arbitrum.io/rpc')
TOKEN = '0x939727d85d99d0ac339bf1b76dfe30ca27c19067'
AMOUNT = Decimal('369.859232953160699920173622889998191251924680922163851546395')

DECIMALS_SIG = '313ce567'

try:
    with httpx.Client(timeout=20) as client:
        resp = client.post(RPC_URL, json={
            'jsonrpc':'2.0','id':1,'method':'eth_call',
            'params':[{'to': TOKEN, 'data': '0x'+DECIMALS_SIG}, 'latest']
        })
    resp.raise_for_status()
    result = resp.json().get('result')
    if result:
        decimals = int(result, 16)
        raw = int(AMOUNT * (Decimal(10) ** decimals))
        print(f'decimals {decimals}')
        print(f'amount_raw {raw}')
    else:
        print(f'Error: No result in JSON response {resp.json()}')
except Exception as e:
    print(f'Error: {e}')

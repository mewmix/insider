import os
import time
import httpx

def rpc_call(url, method, params):
    try:
        t0 = time.time()
        with httpx.Client(timeout=5) as client:
            resp = client.post(
                url,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            )
        dt = time.time() - t0
        if resp.status_code == 200:
            return True, dt, resp.json()
        return False, dt, resp.status_code
    except Exception as e:
        return False, 0, str(e)

rpc_urls = [
    "https://arb1.arbitrum.io/rpc",
    "https://1rpc.io/arb",
    "https://arbitrum.drpc.org",
    "https://arbitrum-one-rpc.publicnode.com",
]

print("Checking RPCs...")
valid_rpcs = []
for url in rpc_urls:
    success, dt, res = rpc_call(url, "eth_blockNumber", [])
    if success:
        print(f"PASS: {url} ({dt:.2f}s)")
        valid_rpcs.append(url)
    else:
        print(f"FAIL: {url} ({res})")

if not valid_rpcs:
    print("ALL RPCs FAILED!")
else:
    print(f"Found {len(valid_rpcs)} working RPCs.")

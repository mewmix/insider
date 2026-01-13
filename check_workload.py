import sqlite3
import os

db_path = "skim_pairs.db"
conn = sqlite3.connect(db_path)

# Load pairs
sql = "SELECT pair_id, token0, token1, dex FROM pairs"
rows = conn.execute(sql).fetchall()

dexes = ["uniswapv2", "camelot", "sushiswapv2", "sushiswap"]
# Normalize dex names
rows_norm = []
for r in rows:
    d = r[3]
    if d == "sushiswap": d = "sushiswapv2"
    rows_norm.append((r[0], r[1], r[2], d))

by_tokens = {}
for r in rows_norm:
    pair_id, t0, t1, dex = r
    key = tuple(sorted([t0.lower(), t1.lower()]))
    if key not in by_tokens:
        by_tokens[key] = set()
    by_tokens[key].add(dex)

count = 0
for k, v in by_tokens.items():
    if len(v) >= 2:
        count += 1

print(f"Total pairs in DB: {len(rows)}")
print(f"Unique token pairs: {len(by_tokens)}")
print(f"Pairs on >= 2 DEXes: {count}")

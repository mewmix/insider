#!/bin/bash
export SKIM_PRIVATE_KEY=4add0adcc0fdea98c1c3ee7dea68cedb5207823153b05d6ba6effa9e1eee225b

python3 scanner.py \
    --loop \
    --db-path skim_pairs.db \
    --triangular \
    --triangular-auto-execute \
    --auto-execute \
    --auto-execute-allow-any \
    --min-net-profit-usd 1.00 \
    --gas-price-gwei 0.02 \
    --triangular-dump opps.jsonl \
    --triangular-simulate-all \
    --triangular-allow-v3 \
    --monstrosity-addr 0x7e5E849D5a3FBAea7044b4b9e47baBb3d6A60283 \
    --aave-pool 0x794a61358D6845594F94dc1DB02A252b5b4814aD \
    --rpc-urls "https://arb1.arbitrum.io/rpc,https://1rpc.io/arb,https://arbitrum.drpc.org" \
    --triangular-safety-bps 10

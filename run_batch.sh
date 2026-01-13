#!/bin/bash
export SKIM_PRIVATE_KEY=4add0adcc0fdea98c1c3ee7dea68cedb5207823153b05d6ba6effa9e1eee225b

# Run batch_exec.py on the opps file in dry-run mode to analyze/verify
# We loop it to continuously analyze new opps that appear in the file
while true; do
    if [ -f opps.jsonl ]; then
        python3 batch_exec.py \
            --opps-file opps.jsonl \
            --monstrosity-addr 0x7e5E849D5a3FBAea7044b4b9e47baBb3d6A60283 \
            --aave-pool 0x794a61358D6845594F94dc1DB02A252b5b4814aD \
            --gas-price-gwei 0.02 \
            --dry-run \
            --auto-execute-allow-any \
            --max 1000
    fi
    sleep 30
done

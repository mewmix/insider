#!/bin/bash

# Kill existing python processes to avoid duplicates
pkill -f "flash_swap_scanner.py" || true

# Narrow/Fast Scan (2-hop)
# Using defaults: 0.02 Gwei, $1 Min Profit, No max pairs.
# Focus on top 25 results, loop, auto-execute.
echo "Starting Narrow Scan..."
python3 flash_swap_scanner.py \
    --loop \
    --auto-execute \
    --max-pairs 0 \
    --gas-price-gwei 0.02 \
    --min-net-profit-usd 1.00 \
    --settle-token weth \
    > narrow_scan.log 2>&1 &

# Wide/Triangular Scan
# Triangular logic, wide search, loop, auto-execute.
echo "Starting Wide Triangular Scan..."
python3 flash_swap_scanner.py \
    --triangular \
    --triangular-auto-execute \
    --loop \
    --auto-execute \
    --max-pairs 0 \
    --gas-price-gwei 0.02 \
    --min-net-profit-usd 1.00 \
    --settle-token weth \
    > wide_scan.log 2>&1 &

echo "Scanners running in background. Check narrow_scan.log and wide_scan.log."

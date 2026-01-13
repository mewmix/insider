#!/bin/bash

# Kill existing python processes to avoid duplicates
pkill -f "universal_scanner.py" || true
pkill -f "flash_swap_scanner.py" || true

# Global Scanner (Universal)
# Auto execute, Hunt profit ($1+), Use DB (skim_pairs.db default)
# Including both 2-hop and Triangular (via --max-hops 3) as per "Wide Hunt" / "Global" implication.
echo "Starting Global Scanner..."
python3 universal_scanner.py \
    --db-path skim_pairs.db \
    --auto-execute \
    --min-net-profit-usd 1.00 \
    --loop \
    --max-hops 3 \
    --settle-token weth \
    --gas-price-gwei 0.02 \
    > global_scan.log 2>&1 &

echo "Global scanner running in background. Check global_scan.log."

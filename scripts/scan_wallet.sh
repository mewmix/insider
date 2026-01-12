#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <wallet_address> [--follow]"
    exit 1
fi

ADDRESS=$1
shift

python3 subgraph_scanner.py --address "$ADDRESS" "$@"

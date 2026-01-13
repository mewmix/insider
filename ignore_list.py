import os
from typing import Set


def parse_ignore_addresses(cli_value: str = "") -> Set[str]:
    sources = [
        os.getenv("IGNORE_ADDRESSES", ""),
        os.getenv("IGNORE_TOKENS", ""),
        os.getenv("IGNORE_PAIRS", ""),
        cli_value or "",
    ]
    ignored = set()
    for source in sources:
        if not source:
            continue
        for item in source.split(","):
            addr = item.strip().lower()
            if addr:
                ignored.add(addr)
    return ignored


def is_ignored_address(address: str, ignored: Set[str]) -> bool:
    return address.lower() in ignored

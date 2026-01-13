import os
from typing import Set


def parse_allow_addresses(cli_value: str = "") -> Set[str]:
    sources = [
        os.getenv("ALLOW_ADDRESSES", ""),
        os.getenv("ALLOW_TOKENS", ""),
        os.getenv("ALLOW_PAIRS", ""),
        cli_value or "",
    ]
    allowed = set()
    for source in sources:
        if not source:
            continue
        for item in source.split(","):
            addr = item.strip().lower()
            if addr:
                allowed.add(addr)
    return allowed


def is_allowed_address(address: str, allowed: Set[str], allow_any: bool) -> bool:
    if not allowed:
        return allow_any
    return address.lower() in allowed

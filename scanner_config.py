import os

RPC_ENDPOINTS = {
    "drpc_ws": "wss://arbitrum.drpc.org",
    "1rpc": "https://1rpc.io/arb",
    "publicnode_ws": "wss://arbitrum-one-rpc.publicnode.com",
    "zan": "https://api.zan.top/arb-one",
    "drpc": "https://arbitrum.drpc.org",
    "fastnode": "https://public-arb-mainnet.fastnode.io",
    "owlracle": "https://rpc.owlracle.info/arb/70d38ce1826c4a60bb2a8e05a6c8b20f",
    "nodies": "https://arbitrum-one-public.nodies.app",
    "publicnode": "https://arbitrum-one-rpc.publicnode.com",
    "tatum": "https://arb-one-mainnet.gateway.tatum.io",
    "tenderly": "https://arbitrum.gateway.tenderly.co",
    "lava": "https://arb1.lava.build",
    "blast": "https://arbitrum-one.public.blastapi.io",
    "subquery": "https://arbitrum.rpc.subquery.network/public",
    "blockpi": "https://arbitrum.public.blockpi.network/v1/rpc/public",
    "pocket": "https://arb-one.api.pocket.network",
    "meowrpc": "https://arbitrum.meowrpc.com",
    "arbitrum": "https://arb1.arbitrum.io/rpc",
    "alchemy": "https://arb-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}",
    "dwellir": "https://api-arbitrum-mainnet-archive.n.dwellir.com/2ccf18bf-2916-4198-8856-42172854353c",
    "poolz": "https://rpc.poolz.finance/arbitrum",
    "onfinality": "https://arbitrum.api.onfinality.io/public",
    "therpc": "https://arbitrum.therpc.io",
    "omniatech": "https://endpoints.omniatech.io/v1/arbitrum/one/public",
    "callstaticrpc_ws": "wss://arbitrum.callstaticrpc.com",
    "stateless": "https://api.stateless.solutions/arbitrum-one/v1/demo",
    "stackup": "https://public.stackup.sh/api/v1/node/arbitrum-one",
    "gatewayfm": "https://rpc.arb1.arbitrum.gateway.fm",
    "unifra": "https://arb-mainnet-public.unifra.io",
    "alchemy_demo": "https://arb-mainnet.g.alchemy.com/v2/demo",
}

def normalize_rpc_url(url: str) -> str:
    alchemy_key = os.getenv("ALCHEMY_API_KEY")
    if "${ALCHEMY_API_KEY}" in url:
        if not alchemy_key:
            return ""
        return url.replace("${ALCHEMY_API_KEY}", alchemy_key)
    return url

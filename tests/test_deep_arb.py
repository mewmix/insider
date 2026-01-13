import pytest
import os
import sys
from decimal import Decimal
from unittest.mock import MagicMock, patch
from web3 import Web3, EthereumTesterProvider
from eth_tester import PyEVMBackend, EthereumTester
from solcx import compile_files, compile_source, install_solc

# Add repo root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scanner import deep_scan_cycles, PairData, execute_monstrosity, ACTION_V2_SWAP, ACTION_AAVE_FLASH
from eth_abi import encode as eth_abi_encode

# Install Solc
install_solc('0.8.20')

@pytest.fixture
def w3():
    # Increase gas limit for complex txs
    backend = PyEVMBackend()
    tester = EthereumTester(backend)
    w3 = Web3(EthereumTesterProvider(tester))
    return w3

@pytest.fixture
def compiled_contracts():
    # Compile Monstrosity and Mocks
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    contracts_path = os.path.join(base_path, "contracts")
    mocks_path = os.path.join(contracts_path, "mocks", "Mocks.sol")
    monstrosity_path = os.path.join(contracts_path, "Monstrosity.sol")

    # We compile both. Assuming imports work relative to contracts_path.
    # We might need to change CWD to contracts/ for solc to find interfaces.
    cwd = os.getcwd()
    os.chdir(contracts_path)
    try:
        compiled = compile_files([mocks_path, monstrosity_path], output_values=['abi', 'bin'], solc_version='0.8.20')
    finally:
        os.chdir(cwd)

    return compiled

def deploy(w3, compiled, contract_name, args=None):
    contract_interface = compiled[contract_name]
    Contract = w3.eth.contract(abi=contract_interface['abi'], bytecode=contract_interface['bin'])
    tx_hash = Contract.constructor(*(args or [])).transact({'from': w3.eth.accounts[0], 'gas': 5000000})
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return w3.eth.contract(address=tx_receipt.contractAddress, abi=contract_interface['abi'])

def test_deep_arb_cycle(w3, compiled_contracts):
    account = w3.eth.accounts[0]

    # 1. Deploy Tokens (WETH + A, B, C, D, E)
    # Using mocks/Mocks.sol:MockERC20
    mock_erc20_key = [k for k in compiled_contracts.keys() if "MockERC20" in k][0]

    weth = deploy(w3, compiled_contracts, mock_erc20_key, ["WETH", "WETH"])
    tokens = {}
    for sym in ["A", "B", "C", "D", "E"]:
        tokens[sym] = deploy(w3, compiled_contracts, mock_erc20_key, [sym, sym])

    # 2. Deploy Pairs
    # WETH->A, A->B, B->C, C->D, D->E, E->WETH
    # Using mocks/Mocks.sol:MockPair
    mock_pair_key = [k for k in compiled_contracts.keys() if "MockPair" in k][0]

    pairs = []
    # (TokenIn, TokenOut, ReserveIn, ReserveOut)
    # We want 10% gain each hop.
    # Reserves: 1000 In, 1100 Out?
    # Price = Out/In = 1.1
    # Actually swap output formula: out = (in * 997 * ReserveOut) / (ReserveIn * 1000 + in * 997)
    # To simplify, we'll set huge reserves to minimize slippage, or just enough.

    cycle_def = [
        ("WETH", "A"),
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),
        ("E", "WETH")
    ]

    pair_objects = {}

    # Fund pairs
    for t_in_sym, t_out_sym in cycle_def:
        t_in = weth if t_in_sym == "WETH" else tokens[t_in_sym]
        t_out = weth if t_out_sym == "WETH" else tokens[t_out_sym]

        pair = deploy(w3, compiled_contracts, mock_pair_key, [t_in.address, t_out.address])
        pair_objects[(t_in_sym, t_out_sym)] = pair

        # Mint to pair
        # Reserve In: 10,000 * 1e18
        # Reserve Out: 11,500 * 1e18 (Price 1.15)

        t_in.functions.mint(pair.address, 10000 * 10**18).transact()
        t_out.functions.mint(pair.address, 11500 * 10**18).transact()
        pair.functions.sync().transact()

        pairs.append(pair)

    # 3. Pathfinding Test
    # Construct 'adj' for deep_scan_cycles
    adj = {}

    def get_pair_data(pair, t0_sym, t1_sym):
        t0 = weth.address if t0_sym == "WETH" else tokens[t0_sym].address
        t1 = weth.address if t1_sym == "WETH" else tokens[t1_sym].address
        return PairData(
            dex="uniswapv2",
            pair_id=pair.address.lower(),
            token0=t0.lower(),
            token1=t1.lower(),
            token0_symbol=t0_sym,
            token1_symbol=t1_sym,
            token0_decimals=18,
            token1_decimals=18,
            reserve0=Decimal(0),
            reserve1=Decimal(0)
        )

    for (t1_sym, t2_sym) in cycle_def:
        t1_addr = (weth.address if t1_sym == "WETH" else tokens[t1_sym].address).lower()
        t2_addr = (weth.address if t2_sym == "WETH" else tokens[t2_sym].address).lower()
        pair = pair_objects[(t1_sym, t2_sym)]
        p_data = get_pair_data(pair, t1_sym, t2_sym)

        adj.setdefault(t1_addr, []).append((t2_addr, p_data))
        adj.setdefault(t2_addr, []).append((t1_addr, p_data)) # Bidirectional

    start_token = weth.address.lower()

    # Scan
    print("Running deep scan...")
    cycles = deep_scan_cycles(adj, start_token, min_hops=4, max_hops=7)

    print(f"Found {len(cycles)} cycles")
    found_6_hop = False
    for c in cycles:
        # Expected: WETH -> A -> B -> C -> D -> E -> WETH
        # c is list of (token_dest, pair)
        # Length should be 6
        if len(c) == 6:
            found_6_hop = True
            path_syms = [p[1].token1_symbol if p[0] == p[1].token1 else p[1].token0_symbol for p in c]
            print("Path:", path_syms)

    assert found_6_hop, "Did not find the 6-hop cycle"

    # 4. Execution Test via Monstrosity
    # Deploy Monstrosity
    monstrosity_key = [k for k in compiled_contracts.keys() if "Monstrosity" in k][0]
    monstrosity = deploy(w3, compiled_contracts, monstrosity_key, [weth.address])

    # Mock Aave Pool
    MockPoolSource = '''
    // SPDX-License-Identifier: MIT
    pragma solidity ^0.8.20;
    interface IFlashLoanReceiver {
        function executeOperation(address[] calldata assets, uint256[] calldata amounts, uint256[] calldata premiums, address initiator, bytes calldata params) external returns (bool);
    }
    import "./interfaces/IERC20.sol";

    contract MockPool {
        function flashLoan(
            address receiverAddress,
            address[] calldata assets,
            uint256[] calldata amounts,
            uint256[] calldata modes,
            address onBehalfOf,
            bytes calldata params,
            uint16 referralCode
        ) external {
            // Transfer funds to receiver
            IERC20(assets[0]).transfer(receiverAddress, amounts[0]);

            // Callback
            uint256[] memory premiums = new uint256[](assets.length);
            IFlashLoanReceiver(receiverAddress).executeOperation(assets, amounts, premiums, msg.sender, params);

            // Take back funds + premium (0 for mock)
            IERC20(assets[0]).transferFrom(receiverAddress, address(this), amounts[0]);
        }
    }
    '''

    # Compile MockPool
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cwd = os.getcwd()
    os.chdir(os.path.join(base_path, "contracts")) # To find interfaces
    try:
        compiled_pool = compile_source(MockPoolSource, output_values=['abi', 'bin'], solc_version='0.8.20')
    finally:
        os.chdir(cwd)

    mock_pool_key = [k for k in compiled_pool.keys() if "MockPool" in k][0]
    mock_pool = deploy(w3, compiled_pool, mock_pool_key)

    # Fund MockPool with WETH
    weth.functions.mint(mock_pool.address, 100 * 10**18).transact()

    # Construct Steps for Monstrosity
    path_pairs = [pair_objects[(s1, s2)] for (s1, s2) in cycle_def]
    path_tokens = [weth.address] + [tokens[s].address for s in ["A", "B", "C", "D", "E"]] + [weth.address]

    steps = []
    amount_in = 1 * 10**18

    for idx, pair in enumerate(path_pairs):
        t_in = path_tokens[idx]
        t_out = path_tokens[idx+1]
        pair_addr = pair.address

        steps.append({
            "action": ACTION_V2_SWAP,
            "target": pair_addr,
            "tokenIn": t_in,
            "tokenOut": t_out,
            "amountIn": amount_in if idx == 0 else 0,
            "minAmountOut": 0, # Simplify for test
            "extraData": b""
        })

    # Encode for Flash Loan
    step_type = "(uint8,address,address,address,uint256,uint256,bytes)"
    encoded_nested = eth_abi_encode(
        [f"{step_type}[]"],
        [[
            (
                s["action"],
                s["target"],
                s["tokenIn"],
                s["tokenOut"],
                s["amountIn"],
                s["minAmountOut"],
                s["extraData"],
            )
            for s in steps
        ]],
    )

    flash_step = {
        "action": ACTION_AAVE_FLASH,
        "target": mock_pool.address,
        "tokenIn": weth.address,
        "tokenOut": weth.address,
        "amountIn": amount_in,
        "minAmountOut": 0,
        "extraData": encoded_nested
    }

    print("Executing Monstrosity Trade on Tester...")

    formatted_steps = [(
        flash_step["action"],
        flash_step["target"],
        flash_step["tokenIn"],
        flash_step["tokenOut"],
        flash_step["amountIn"],
        flash_step["minAmountOut"],
        flash_step["extraData"]
    )]

    tx_hash = monstrosity.functions.execute(formatted_steps, 0).transact({'from': account})
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    assert receipt.status == 1, "Transaction failed"
    print("Trade executed successfully!")

@patch('scanner.fetch_reserves_with_rotation')
@patch('scanner.token_price_usd')
def test_complex_scan_smoke(mock_price, mock_fetch, w3):
    # Smoke test for complex_scan logic integration
    from scanner import complex_scan, PairData

    mock_fetch.return_value = (1000 * 10**18, 1000 * 10**18)
    mock_price.return_value = Decimal("2000") # WETH Price

    # Construct minimal data
    p1 = PairData("uniswapv2", "0x1", "0xa", "0xb", "A", "B", 18, 18, Decimal(0), Decimal(0))
    p2 = PairData("uniswapv2", "0x2", "0xb", "0xc", "B", "C", 18, 18, Decimal(0), Decimal(0))
    p3 = PairData("uniswapv2", "0x3", "0xc", "0xa", "C", "A", 18, 18, Decimal(0), Decimal(0))

    pairs = [p1, p2, p3]
    pair_index = {
        tuple(sorted(["0xa", "0xb"])): [p1],
        tuple(sorted(["0xb", "0xc"])): [p2],
        tuple(sorted(["0xc", "0xa"])): [p3]
    }

    # Run scan
    # It should not crash
    complex_scan(
        pairs=pairs,
        pair_index=pair_index,
        rpc_urls=["http://mock"],
        start_idx=0,
        reserve_cache={},
        weth_price=Decimal("2000"),
        gas_units=500000,
        gas_price_gwei=Decimal("0.1"),
        min_net_profit_usd=Decimal("0"),
        fee_by_dex={},
        max_trade_frac=Decimal("1.0"),
        auto_execute=False,
        safety_bps=Decimal("0"),
        monstrosity_address="0x0",
        aave_pool_address="0x0",
        ignore_addresses=set(),
        dump_path="",
        allow_addresses=set(),
        allow_any=True,
        min_pair_liquidity_usd=Decimal("0"),
        simulate_all=False,
        allow_v3=False,
        v3_amount_in=Decimal("0"),
        min_hops=3,
        max_hops=3
    )

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))

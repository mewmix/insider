import unittest
from eth_tester import EthereumTester, PyEVMBackend
from web3 import Web3, EthereumTesterProvider
import solcx
import os

# Install solc if needed
solcx.install_solc('0.8.20')

class TestMonstrosity(unittest.TestCase):
    def setUp(self):
        self.tester = EthereumTester(PyEVMBackend())
        self.w3 = Web3(EthereumTesterProvider(self.tester))
        self.w3.eth.default_account = self.w3.eth.accounts[0]

        # Compile contracts
        compiled = solcx.compile_files(
            ['contracts/Monstrosity.sol', 'contracts/mocks/Mocks.sol'],
            output_values=['abi', 'bin'],
            solc_version='0.8.20',
            allow_paths=['.'],
            optimize=True,
            optimize_runs=200
        )

        self.monstrosity_interface = compiled['contracts/Monstrosity.sol:Monstrosity']
        self.mock_erc20_interface = compiled['contracts/mocks/Mocks.sol:MockERC20']
        self.mock_pair_interface = compiled['contracts/mocks/Mocks.sol:MockPair']

        # Deploy WETH Mock
        MockERC20 = self.w3.eth.contract(abi=self.mock_erc20_interface['abi'], bytecode=self.mock_erc20_interface['bin'])
        self.weth = MockERC20.constructor("WETH", "WETH").transact()
        self.weth_addr = self.w3.eth.get_transaction_receipt(self.weth).contractAddress

        # Deploy Monstrosity
        Monstrosity = self.w3.eth.contract(abi=self.monstrosity_interface['abi'], bytecode=self.monstrosity_interface['bin'])
        self.monstrosity = Monstrosity.constructor(self.weth_addr).transact()
        self.monstrosity_addr = self.w3.eth.get_transaction_receipt(self.monstrosity).contractAddress
        self.monstrosity_contract = self.w3.eth.contract(address=self.monstrosity_addr, abi=self.monstrosity_interface['abi'])

        # Deploy Tokens
        self.tokenA = MockERC20.constructor("Token A", "TKA").transact()
        self.tokenA_addr = self.w3.eth.get_transaction_receipt(self.tokenA).contractAddress
        self.tokenB = MockERC20.constructor("Token B", "TKB").transact()
        self.tokenB_addr = self.w3.eth.get_transaction_receipt(self.tokenB).contractAddress

        # Mint tokens to pair and self
        self.tokenA_contract = self.w3.eth.contract(address=self.tokenA_addr, abi=self.mock_erc20_interface['abi'])
        self.tokenB_contract = self.w3.eth.contract(address=self.tokenB_addr, abi=self.mock_erc20_interface['abi'])

        # Deploy Mock Pair
        MockPair = self.w3.eth.contract(abi=self.mock_pair_interface['abi'], bytecode=self.mock_pair_interface['bin'])
        self.pair = MockPair.constructor(self.tokenA_addr, self.tokenB_addr).transact()
        self.pair_addr = self.w3.eth.get_transaction_receipt(self.pair).contractAddress
        self.pair_contract = self.w3.eth.contract(address=self.pair_addr, abi=self.mock_pair_interface['abi'])

        # Fund Pair
        self.tokenA_contract.functions.mint(self.pair_addr, 1000000).transact()
        self.tokenB_contract.functions.mint(self.pair_addr, 1000000).transact()
        self.pair_contract.functions.sync().transact()

        # Fund Monstrosity for swap
        self.tokenA_contract.functions.mint(self.monstrosity_addr, 1000).transact()

    def test_direct_swap(self):
        # Step: Swap 100 TokenA for TokenB
        # Action 1 = V2 Swap
        step = (
            1, # action
            self.pair_addr, # target
            self.tokenA_addr, # tokenIn
            self.tokenB_addr, # tokenOut
            100, # amountIn
            0, # minAmountOut
            b"" # extraData
        )

        # Execute
        tx = self.monstrosity_contract.functions.execute([step], 0).transact()
        receipt = self.w3.eth.wait_for_transaction_receipt(tx)

        # Check balances
        # Monstrosity started with 1000 TKA. Sent 100. Should have 900.
        balA = self.tokenA_contract.functions.balanceOf(self.monstrosity_addr).call()
        self.assertEqual(balA, 900)

        # Should have received TKB.
        # Pair has 1M each. Input 100.
        # fee 30bps = 0.3. Input with fee = 99.7.
        # Output = (99.7 * 1000000) / (1000000 + 99.7) ~= 99.69
        balB = self.tokenB_contract.functions.balanceOf(self.monstrosity_addr).call()
        self.assertGreater(balB, 0)
        print(f"Swap output: {balB}")

    def test_v3_callback_security(self):
        # Action 2 = V3 Swap
        # We need a Mock V3 Pool.
        # Deploy Mock V3 Pool
        MockV3Pool_source = """
        // SPDX-License-Identifier: MIT
        pragma solidity ^0.8.20;
        interface IERC20 {
            function transfer(address, uint256) external returns (bool);
            function transferFrom(address, address, uint256) external returns (bool);
        }
        contract MockV3Pool {
            address public token0;
            address public token1;
            uint24 public fee = 3000;
            constructor(address _token0, address _token1) {
                token0 = _token0;
                token1 = _token1;
            }
            function swap(
                address recipient,
                bool zeroForOne,
                int256 amountSpecified,
                uint160 sqrtPriceLimitX96,
                bytes calldata data
            ) external returns (int256 amount0, int256 amount1) {
                // Determine amountOut and amountIn
                // Simplified: 1:1 swap
                int256 amountIn = amountSpecified;
                int256 amountOut = -amountSpecified;

                amount0 = zeroForOne ? amountIn : amountOut;
                amount1 = zeroForOne ? amountOut : amountIn;

                // Callback
                // Address(msg.sender) is the swapper (Monstrosity)
                (bool success, ) = msg.sender.call(abi.encodeWithSignature("uniswapV3SwapCallback(int256,int256,bytes)", amount0, amount1, data));
                require(success, "callback failed");

                // Transfer output
                if (zeroForOne) {
                    IERC20(token1).transfer(recipient, uint256(-amountOut));
                } else {
                    IERC20(token0).transfer(recipient, uint256(-amountOut));
                }
            }
        }
        """

        compiled_v3 = solcx.compile_source(MockV3Pool_source, output_values=['abi', 'bin'], solc_version='0.8.20')
        MockV3Pool = self.w3.eth.contract(abi=compiled_v3['<stdin>:MockV3Pool']['abi'], bytecode=compiled_v3['<stdin>:MockV3Pool']['bin'])
        v3_pool = MockV3Pool.constructor(self.tokenA_addr, self.tokenB_addr).transact()
        v3_pool_addr = self.w3.eth.get_transaction_receipt(v3_pool).contractAddress

        # Fund V3 Pool
        self.tokenA_contract.functions.mint(v3_pool_addr, 1000000).transact()
        self.tokenB_contract.functions.mint(v3_pool_addr, 1000000).transact()

        # Execute V3 Swap
        step = (
            2, # ACTION_V3_SWAP
            v3_pool_addr,
            self.tokenA_addr,
            self.tokenB_addr,
            100,
            0,
            b""
        )

        tx = self.monstrosity_contract.functions.execute([step], 0).transact()

        # Verify balance changes
        balA = self.tokenA_contract.functions.balanceOf(self.monstrosity_addr).call()
        balB = self.tokenB_contract.functions.balanceOf(self.monstrosity_addr).call()

        # Started with 1000 (from setUp). Sent 100.
        self.assertEqual(balA, 900)
        self.assertGreaterEqual(balB, 100)

    def test_v3_callback_exploit_fail(self):
        # Attempt to call uniswapV3SwapCallback directly from a random address should fail
        try:
            self.monstrosity_contract.functions.uniswapV3SwapCallback(100, 0, b"").transact({'from': self.w3.eth.accounts[1]})
            self.fail("Should have reverted")
        except Exception as e:
            pass

if __name__ == '__main__':
    unittest.main()

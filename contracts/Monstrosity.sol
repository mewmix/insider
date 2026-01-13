// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./interfaces/IERC20.sol";
import "./interfaces/IUniswapV2Pair.sol";
import "./interfaces/IUniswapV3Pool.sol";
import "./interfaces/IPool.sol";
import "./interfaces/ICurvePool.sol";
import "./interfaces/IComet.sol";
import "./interfaces/IFlashLoanReceiver.sol";
import "./interfaces/IWETH.sol";

contract Monstrosity is IFlashLoanReceiver {
    address public immutable owner;
    address public immutable WETH;

    // Action Types
    uint8 constant ACTION_V2_SWAP = 1;
    uint8 constant ACTION_V3_SWAP = 2;
    uint8 constant ACTION_AAVE_FLASH = 3;
    uint8 constant ACTION_V2_FLASH_SWAP = 4;
    uint8 constant ACTION_CURVE_SWAP = 5;
    uint8 constant ACTION_COMPOUND_V3_SUPPLY = 6;
    uint8 constant ACTION_COMPOUND_V3_WITHDRAW = 7;

    struct Step {
        uint8 action;
        address target; // Pool or Pair address
        address tokenIn;
        address tokenOut;
        uint256 amountIn; // If 0, use entire balance of tokenIn
        uint256 minAmountOut;
        bytes extraData; // For V3: zeroForOne, sqrtPriceLimitX96. For Aave: assets/amounts if initiating nested?
    }

    // Transient-like storage for callback security
    address private activeV3Pool;

    constructor(address _weth) {
        owner = msg.sender;
        WETH = _weth;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    receive() external payable {}

    // Main entry point
    function execute(Step[] calldata steps, uint256 minProfitUSD) external onlyOwner {
        uint256 startBal = IERC20(WETH).balanceOf(address(this));

        _runSteps(steps);

        uint256 endBal = IERC20(WETH).balanceOf(address(this));
        require(endBal >= startBal + minProfitUSD, "insufficient profit");
    }

    function _runSteps(Step[] memory steps) internal {
        for (uint256 i = 0; i < steps.length; i++) {
            Step memory step = steps[i];

            uint256 amountIn = step.amountIn;
            if (amountIn == 0) {
                amountIn = IERC20(step.tokenIn).balanceOf(address(this));
            }
            require(amountIn > 0, "zero amount in");

            if (step.action == ACTION_V2_SWAP) {
                _swapV2(step, amountIn);
            } else if (step.action == ACTION_V3_SWAP) {
                _swapV3(step, amountIn);
            } else if (step.action == ACTION_CURVE_SWAP) {
                _swapCurve(step, amountIn);
            } else if (step.action == ACTION_AAVE_FLASH) {
                _flashAave(step, amountIn);
            } else if (step.action == ACTION_V2_FLASH_SWAP) {
                _flashV2(step, amountIn);
            } else if (step.action == ACTION_COMPOUND_V3_SUPPLY) {
                _compoundSupply(step, amountIn);
            } else if (step.action == ACTION_COMPOUND_V3_WITHDRAW) {
                _compoundWithdraw(step, amountIn);
            }
        }
    }

    function _flashAave(Step memory step, uint256 amountIn) internal {
        // extraData contains encoded nested steps
        // we must not pass amountIn as it is irrelevant for initiator usually?
        // Actually, amountIn IS the flash amount.

        address[] memory assets = new address[](1);
        assets[0] = step.tokenIn;
        uint256[] memory amounts = new uint256[](1);
        amounts[0] = amountIn;

        // Pass nested steps as params
        // step.extraData MUST be abi.encode(Step[] steps)
        // But wait, extraData is just bytes. The caller must prepare this.
        uint256[] memory modes = new uint256[](1); // 0 = no debt

        IPool(step.target).flashLoan(
            address(this),
            assets,
            amounts,
            modes,
            address(this),
            step.extraData, // contains nested steps
            0
        );
    }

    function _flashV2(Step memory step, uint256 amountIn) internal {
        // V2 Flash Swap Initiation
        // tokenIn is the token to BORROW
        // target is the pair

        IUniswapV2Pair pair = IUniswapV2Pair(step.target);
        address token0 = pair.token0();
        address token1 = pair.token1();

        uint256 amount0 = step.tokenIn == token0 ? amountIn : 0;
        uint256 amount1 = step.tokenIn == token0 ? 0 : amountIn;

        // Data needs to encode the nested steps
        // AND potentially the fee info if we want to be safe, or just rely on steps.
        // We pass step.extraData which contains nested Step[]

        // We need to encode it properly for uniswapV2Call to decode.
        // uniswapV2Call expects (Step[], address) where address is just logic??
        // No, in my previous code: (Step[] memory steps, address pair) = abi.decode(data, (Step[], address));
        // Why pass pair? The callback validates msg.sender.
        // Let's just pass the steps.

        pair.swap(amount0, amount1, address(this), step.extraData);
    }

    function _compoundSupply(Step memory step, uint256 amountIn) internal {
        // Supply to Compound V3 (Comet)
        // target is Comet address
        // tokenIn is asset to supply
        IERC20(step.tokenIn).approve(step.target, amountIn);
        IComet(step.target).supply(step.tokenIn, amountIn);
    }

    function _compoundWithdraw(Step memory step, uint256 amountIn) internal {
        // Withdraw from Compound V3
        // amountIn is amount to withdraw
        IComet(step.target).withdraw(step.tokenOut, amountIn);
    }

    function _swapCurve(Step memory step, uint256 amountIn) internal {
        // Curve Exchange
        // extraData must contain i (int128) and j (int128)
        (int128 i, int128 j) = abi.decode(step.extraData, (int128, int128));

        IERC20(step.tokenIn).approve(step.target, amountIn);
        try ICurvePool(step.target).exchange(i, j, amountIn, step.minAmountOut) {
            // Success
        } catch {
             // Try uint256 interface if int128 fails (some pools differ)
             ICurvePool(step.target).exchange(uint256(int256(i)), uint256(int256(j)), amountIn, step.minAmountOut);
        }
    }

    function _swapV2(Step memory step, uint256 amountIn) internal {
        IUniswapV2Pair pair = IUniswapV2Pair(step.target);
        address token0 = pair.token0();
        address token1 = pair.token1();

        require(step.tokenIn == token0 || step.tokenIn == token1, "invalid token in");

        // Transfer tokens to pair
        IERC20(step.tokenIn).transfer(step.target, amountIn);

        (uint112 r0, uint112 r1, ) = pair.getReserves();
        uint256 reserveIn = step.tokenIn == token0 ? r0 : r1;
        uint256 reserveOut = step.tokenIn == token0 ? r1 : r0;

        // Fee is usually 30bps (0.3%) for V2, but Camelot/Sushi might differ.
        // We calculate amountOut based on reserves.
        // Ideally, we pass expected amountOut or calculate it on chain.
        // For flexibility, let's assume we want to support any fee.
        // But getAmountOut requires fee.
        // We can pass fee in extraData or just calculate output needed.
        // Or, simplified: we let the caller specify minAmountOut and we try to get that?
        // No, standard V2 swap requires specifying `amountOut` in the call.

        // Let's decode fee from extraData or assume 30bps.
        uint256 feeBps = 30;
        if (step.extraData.length > 0) {
             feeBps = abi.decode(step.extraData, (uint256));
        }

        uint256 amountOut;
        {
            uint256 amountInWithFee = amountIn * (10000 - feeBps);
            uint256 numerator = amountInWithFee * reserveOut;
            uint256 denominator = (reserveIn * 10000) + amountInWithFee;
            amountOut = numerator / denominator;
        }

        require(amountOut >= step.minAmountOut, "insufficient output V2");

        pair.swap(
            step.tokenIn == token0 ? 0 : amountOut,
            step.tokenIn == token0 ? amountOut : 0,
            address(this),
            ""
        );
    }

    function _swapV3(Step memory step, uint256 amountIn) internal {
        // V3 swap logic
        IUniswapV3Pool pool = IUniswapV3Pool(step.target);
        address token0 = pool.token0();
        // address token1 = pool.token1();
        bool zeroForOne = step.tokenIn == token0;

        int256 amountSpecified = int256(amountIn);
        uint160 sqrtPriceLimitX96 = 0; // No limit
        if (step.extraData.length >= 32) {
             sqrtPriceLimitX96 = abi.decode(step.extraData, (uint160));
        }
        if (sqrtPriceLimitX96 == 0) {
             sqrtPriceLimitX96 = zeroForOne ? 4295128739 + 1 : 1461446703485210103287273052203988822378723970342 - 1;
        }

        activeV3Pool = step.target; // Set expected callback caller
        pool.swap(
            address(this),
            zeroForOne,
            amountSpecified,
            sqrtPriceLimitX96,
            abi.encode(step.tokenIn)
        );
        activeV3Pool = address(0); // Clear after swap
    }

    function uniswapV3SwapCallback(
        int256 amount0Delta,
        int256 amount1Delta,
        bytes calldata /* data */
    ) external {
        require(msg.sender == activeV3Pool, "invalid callback caller");
        require(amount0Delta > 0 || amount1Delta > 0, "invalid delta");

        // Simplified: just pay what is positive
        if (amount0Delta > 0) {
            address token0 = IUniswapV3Pool(msg.sender).token0();
            IERC20(token0).transfer(msg.sender, uint256(amount0Delta));
        }
        if (amount1Delta > 0) {
             address token1 = IUniswapV3Pool(msg.sender).token1();
             IERC20(token1).transfer(msg.sender, uint256(amount1Delta));
        }
    }

    // Aave V3 Flash Loan Receiver
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external returns (bool) {
        require(initiator == address(this), "untrusted initiator");

        // Decode steps and run them
        Step[] memory steps = abi.decode(params, (Step[]));
        _runSteps(steps);

        // Approve repayment
        for (uint256 i = 0; i < assets.length; i++) {
            uint256 amountOwing = amounts[i] + premiums[i];
            IERC20(assets[i]).approve(msg.sender, amountOwing);
        }

        return true;
    }

    // Aave Flash Loan Initiator
    function flashLoanAave(
        address pool,
        address[] calldata assets,
        uint256[] calldata amounts,
        Step[] calldata steps
    ) external onlyOwner {
        // Encode steps to pass to callback
        bytes memory params = abi.encode(steps);
        uint256[] memory modes = new uint256[](assets.length); // 0 = no debt

        IPool(pool).flashLoan(
            address(this),
            assets,
            amounts,
            modes,
            address(this),
            params,
            0
        );
    }

    // Uniswap V2 Flash Swap Initiator
    // We can use execute() to call a V2 pair with swap(..., data)
    // The callback uniswapV2Call will handle the rest.

    function uniswapV2Call(address sender, uint256 amount0, uint256 amount1, bytes calldata data) external {
        require(sender == address(this), "sender must be this");

        // If we encoded just Step[], decode just Step[]
        Step[] memory steps = abi.decode(data, (Step[]));

        // Execute steps
        _runSteps(steps);

        uint256 amountBorrowed = amount0 > 0 ? amount0 : amount1;

        // Repayment
        // If 0.3% is standard, we use 997.
        // If we want dynamic fee, we could encode it in data.
        // But for "Monstrosity" we want generic.
        // We'll stick to 0.3% for now as standard V2.
        // If user wants custom fee, they must ensure they send enough back or we can add logic later.

        uint256 feeNumerator = amountBorrowed * 1000;
        uint256 repayment = (feeNumerator / 997) + 1;

        address tokenBorrowed = amount0 > 0 ? IUniswapV2Pair(msg.sender).token0() : IUniswapV2Pair(msg.sender).token1();
        IERC20(tokenBorrowed).transfer(msg.sender, repayment);
    }

    function withdraw(address token) external onlyOwner {
        IERC20(token).transfer(owner, IERC20(token).balanceOf(address(this)));
    }

    function withdrawETH() external onlyOwner {
        payable(owner).transfer(address(this).balance);
    }
}

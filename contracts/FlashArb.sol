// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IUniswapV2Pair {
    function token0() external view returns (address);
    function token1() external view returns (address);
    function getReserves() external view returns (uint112, uint112, uint32);
    function swap(uint256, uint256, address, bytes calldata) external;
}

interface IERC20 {
    function balanceOf(address) external view returns (uint256);
    function transfer(address, uint256) external returns (bool);
}

contract FlashArb {
    struct CallbackData {
        address pairBorrow;
        address pairSwap;
        address tokenBorrow;
        uint256 amountBorrow;
        uint256 feeBorrowBps;
        uint256 feeSwapBps;
        uint256 minProfit;
    }

    function execute(
        address pairBorrow,
        address pairSwap,
        address tokenBorrow,
        uint256 amountBorrow,
        uint256 feeBorrowBps,
        uint256 feeSwapBps,
        uint256 minProfit
    ) external {
        IUniswapV2Pair pair = IUniswapV2Pair(pairBorrow);
        address token0 = pair.token0();
        address token1 = pair.token1();
        bytes memory data = abi.encode(
            CallbackData({
                pairBorrow: pairBorrow,
                pairSwap: pairSwap,
                tokenBorrow: tokenBorrow,
                amountBorrow: amountBorrow,
                feeBorrowBps: feeBorrowBps,
                feeSwapBps: feeSwapBps,
                minProfit: minProfit
            })
        );
        if (tokenBorrow == token0) {
            pair.swap(amountBorrow, 0, address(this), data);
        } else if (tokenBorrow == token1) {
            pair.swap(0, amountBorrow, address(this), data);
        } else {
            revert("borrow token not in pair");
        }
    }

    function uniswapV2Call(address, uint256 amount0, uint256 amount1, bytes calldata data) external {
        _handleCallback(amount0, amount1, data);
    }

    function _handleCallback(uint256 amount0, uint256 amount1, bytes calldata data) internal {
        CallbackData memory decoded = abi.decode(data, (CallbackData));
        IUniswapV2Pair borrowPair = IUniswapV2Pair(decoded.pairBorrow);
        require(msg.sender == decoded.pairBorrow, "invalid callback");

        address token0 = borrowPair.token0();
        address token1 = borrowPair.token1();
        uint256 amountBorrowed = amount0 > 0 ? amount0 : amount1;
        require(amountBorrowed == decoded.amountBorrow, "unexpected borrow");

        (uint112 r0, uint112 r1, ) = borrowPair.getReserves();
        uint256 reserve0 = uint256(r0);
        uint256 reserve1 = uint256(r1);

        address tokenBorrow = decoded.tokenBorrow;
        address tokenPay = tokenBorrow == token0 ? token1 : token0;

        // Swap borrowed token on the second pair to get the pay token.
        IUniswapV2Pair swapPair = IUniswapV2Pair(decoded.pairSwap);
        address swapToken0 = swapPair.token0();
        address swapToken1 = swapPair.token1();

        bool borrowIsSwapToken0 = tokenBorrow == swapToken0;
        require(borrowIsSwapToken0 || tokenBorrow == swapToken1, "swap token mismatch");

        (uint112 sr0, uint112 sr1, ) = swapPair.getReserves();
        uint256 swapReserveIn = borrowIsSwapToken0 ? uint256(sr0) : uint256(sr1);
        uint256 swapReserveOut = borrowIsSwapToken0 ? uint256(sr1) : uint256(sr0);

        uint256 amountOut = getAmountOut(amountBorrowed, swapReserveIn, swapReserveOut, decoded.feeSwapBps);
        IERC20(tokenBorrow).transfer(decoded.pairSwap, amountBorrowed);
        if (borrowIsSwapToken0) {
            swapPair.swap(0, amountOut, address(this), "");
        } else {
            swapPair.swap(amountOut, 0, address(this), "");
        }

        uint256 amountRequired;
        if (tokenBorrow == token0) {
            amountRequired = getAmountIn(amountBorrowed, reserve1, reserve0, decoded.feeBorrowBps);
        } else {
            amountRequired = getAmountIn(amountBorrowed, reserve0, reserve1, decoded.feeBorrowBps);
        }

        uint256 balancePay = IERC20(tokenPay).balanceOf(address(this));
        require(balancePay >= amountRequired + decoded.minProfit, "insufficient profit");
        IERC20(tokenPay).transfer(decoded.pairBorrow, amountRequired);
    }

    function getAmountOut(
        uint256 amountIn,
        uint256 reserveIn,
        uint256 reserveOut,
        uint256 feeBps
    ) public pure returns (uint256) {
        require(amountIn > 0 && reserveIn > 0 && reserveOut > 0, "invalid");
        uint256 amountInWithFee = amountIn * (10000 - feeBps);
        uint256 numerator = amountInWithFee * reserveOut;
        uint256 denominator = reserveIn * 10000 + amountInWithFee;
        return numerator / denominator;
    }

    function getAmountIn(
        uint256 amountOut,
        uint256 reserveIn,
        uint256 reserveOut,
        uint256 feeBps
    ) public pure returns (uint256) {
        require(amountOut > 0 && reserveIn > 0 && reserveOut > amountOut, "invalid");
        uint256 numerator = reserveIn * amountOut * 10000;
        uint256 denominator = (reserveOut - amountOut) * (10000 - feeBps);
        return (numerator / denominator) + 1;
    }
}

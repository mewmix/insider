// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IComet {
    function supply(address asset, uint amount) external;
    function withdraw(address asset, uint amount) external;
    function baseToken() external view returns (address);
    function allow(address manager, bool isAllowed) external;
    function borrow(address asset, uint amount) external;
    function hasPermission(address owner, address manager) external view returns (bool);
}

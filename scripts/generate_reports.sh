#!/bin/bash

echo "Generating Active Positions Report..."
python3 active_positions_report.py --with-metadata

echo -e "\nGenerating PnL Leaderboard..."
python3 pnl_leaderboard.py > top_pnl_accounts.txt

echo -e "\nReports generated."


#!/bin/bash
set -e

echo "[*] Stopping existing server (if any)..."
pkill -f "python main.py --server" || true

echo "[*] Starting new server..."
nohup python main.py --server --host 0.0.0.0 --port 8000 > agent.log 2>&1 & echo $! > agent.pid

echo "[*] Done. PID stored in agent.pid"

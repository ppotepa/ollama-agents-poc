#!/bin/bash
set -e

echo "[*] Stopping existing uvicorn server (if any)..."
pkill -f "uvicorn server:app" || true

echo "[*] Starting new uvicorn server..."
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > agent.log 2>&1 & echo $! > agent.pid

echo "[*] Done. PID stored in agent.pid"

#!/bin/bash
set -e

echo "[*] Stopping existing server (if any)..."
pkill -f "python main.py --server" || true

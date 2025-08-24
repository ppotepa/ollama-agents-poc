#!/bin/bash
set -e

echo "[*] Stopping existing uvicorn server (if any)..."
pkill -f "uvicorn server:app" || true

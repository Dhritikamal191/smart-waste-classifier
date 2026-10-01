#!/bin/sh

set -e

echo "=========================================="
echo " Smart Waste Classifier API"
echo "=========================================="

echo "Starting FastAPI server..."

exec uvicorn src.api:app \
    --host 0.0.0.0 \
    --port 8000
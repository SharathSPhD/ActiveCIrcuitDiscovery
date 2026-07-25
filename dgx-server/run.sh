#!/usr/bin/env bash
# Start the ACD live-demo backend on the DGX Spark.
# Usage: ./run.sh [gemma|llama]  (default gemma)
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

MODEL="${1:-gemma}"
export ACD_API_KEY="${ACD_API_KEY:?set ACD_API_KEY first, e.g. export ACD_API_KEY=$(openssl rand -hex 16)}"

source .venv/bin/activate 2>/dev/null || true
pip install -q -r dgx-server/requirements.txt

python dgx-server/server.py --model "$MODEL" --precompute --record --port 8787

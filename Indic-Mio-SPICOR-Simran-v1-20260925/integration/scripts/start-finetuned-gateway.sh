#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -x "$PROJECT_ROOT/.venv/bin/uvicorn" ]]; then
  echo "Gateway environment is missing. Create .venv and install requirements.package.txt." >&2
  exit 1
fi

export MIOTTS_MODEL="indic-mio-spicor-fullshort-stage2"
exec "$PROJECT_ROOT/.venv/bin/uvicorn" app.main:app --host 127.0.0.1 --port 8080


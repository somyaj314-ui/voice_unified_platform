#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export MIOTTS_MODEL_PATH="$PROJECT_ROOT/models/Indic-Mio"
export MIOTTS_LORA_PATH="$PROJECT_ROOT/finetuning/runs/indic-mio-spicor-fullshort-stage2"
export MIOTTS_LLM_MODEL="indic-mio-spicor-fullshort-stage2"
# This miocodec version requires a Hugging Face repository ID. It cannot load
# the same files by passing a local directory path.
export MIOTTS_CODEC_MODEL="Aratako/MioCodec-25Hz-44.1kHz-v2"

exec bash "$PROJECT_ROOT/scripts/start-backend.sh"


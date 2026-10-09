#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

resolve_local_path() {
  local value="$1"
  if [[ -n "$value" && "$value" != /* && -e "$PROJECT_ROOT/$value" ]]; then
    printf '%s\n' "$PROJECT_ROOT/$value"
  else
    printf '%s\n' "$value"
  fi
}

MODEL_PATH="${MIOTTS_MODEL_PATH:-models/Indic-Mio}"
LORA_PATH="${MIOTTS_LORA_PATH:-}"
SERVED_MODEL="${MIOTTS_LLM_MODEL:-$MODEL_PATH}"
CODEC_MODEL="${MIOTTS_CODEC_MODEL:-Aratako/MioCodec-25Hz-44.1kHz-v2}"
LOG_DIR="${MIOTTS_LOG_DIR:-logs}"

# vLLM is launched before changing directories, but resolving the model and
# adapter here also makes their locations independent of the caller's cwd.
# MioCodec's current from_pretrained implementation requires a Hugging Face
# repository ID, so CODEC_MODEL must remain a repo ID rather than a local path.
MODEL_PATH="$(resolve_local_path "$MODEL_PATH")"
LORA_PATH="$(resolve_local_path "$LORA_PATH")"

mkdir -p "$LOG_DIR"
VLLM_LOG="$LOG_DIR/vllm.log"

# Interrupted/failed vLLM launches can leave the API server or EngineCore child
# alive. They retain GPU memory and keep appending old errors to the shared log.
if pgrep -f '[v]llm.*Indic-Mio|[E]ngineCore' >/dev/null 2>&1; then
  echo "Stopping stale Indic-Mio vLLM processes from earlier runs..."
  pkill -TERM -f '[v]llm.*Indic-Mio|[E]ngineCore' 2>/dev/null || true
  sleep 3
  pkill -KILL -f '[v]llm.*Indic-Mio|[E]ngineCore' 2>/dev/null || true
fi
: >"$VLLM_LOG"

# vLLM's V2 GPU runner requires CUDA UVA/pinned host memory. NVIDIA's WSL2
# passthrough currently reports UVA unavailable on consumer Blackwell cards.
# The V1 runner has the same inference API without that always-on UVA buffer.
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-600}"
# FlashInfer's sampler JIT requires a full CUDA toolkit/nvcc. The native
# vLLM sampler is compatible with the driver-only CUDA runtime shipped by WSL.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

VLLM_LORA_ARGS=()
if [[ -n "$LORA_PATH" ]]; then
  if [[ ! -d "$LORA_PATH" ]]; then
    echo "LoRA adapter directory not found: $LORA_PATH" >&2
    exit 1
  fi
  VLLM_LORA_ARGS=(--enable-lora --lora-modules "$SERVED_MODEL=$LORA_PATH")
  echo "Serving LoRA adapter '$SERVED_MODEL' from $LORA_PATH"
fi

.venv-vllm/bin/vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-1024}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.55}" \
  --enforce-eager \
  "${VLLM_LORA_ARGS[@]}" \
  >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT INT TERM

echo "Starting Indic-Mio vLLM (PID $VLLM_PID); log: $VLLM_LOG"
VLLM_READY=0
for second in {1..600}; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited before becoming ready. Last log lines:" >&2
    tail -n 120 "$VLLM_LOG" >&2
    exit 1
  fi
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    VLLM_READY=1
    break
  fi
  if (( second % 30 == 0 )); then
    echo "Still initializing vLLM (${second}s elapsed)..."
    tail -n 3 "$VLLM_LOG" | sed 's/^/  /'
  fi
  sleep 1
done

if [[ "$VLLM_READY" != "1" ]]; then
  echo "vLLM did not become ready within 600 seconds. Last log lines:" >&2
  tail -n 120 "$VLLM_LOG" >&2
  exit 1
fi

echo "vLLM is ready; starting MioCodec API on http://127.0.0.1:8001"

cd "$PROJECT_ROOT/vendor/MioTTS-Inference"
exec .venv/bin/python run_server.py \
  --host 127.0.0.1 \
  --port 8001 \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --llm-model "$SERVED_MODEL" \
  --codec-model "$CODEC_MODEL" \
  --allowed-audio-exts ".wav,.flac,.ogg,.mp3,.m4a" \
  --device cuda

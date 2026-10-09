# Indic-Mio SPICOR Simran v1

This is the deployable handoff package for the experimental Indic-Mio LoRA
adapter selected in the MOSS-TTS project.

## What is included

- `adapter/`: final PEFT LoRA adapter and matching tokenizer files.
- `voices/simran_clone_indian.mp3`: production Simran reference audio.
- `config/production-config.json`: exact generation and runtime settings.
- `config/voices.json`: voice registration entry.
- `integration/`: the gateway and launch helpers used in the source project.
- `TRAINING-NOTES.md`: dataset split and training details.
- `LICENSE-NOTICE.md`: model, dataset, and voice-rights notes.

Training checkpoints and optimizer states are intentionally excluded. They are
not needed for inference and would increase the archive by roughly 230 MB.

## Required external components

This ZIP is not a standalone full model. Obtain these separately:

1. Base model: `SPRINGLab/Indic-Mio`
2. Codec: `Aratako/MioCodec-25Hz-44.1kHz-v2` (downloaded/cached by its
   Hugging Face repository ID; the current loader does not accept a local path)
3. MioTTS-Inference and a vLLM environment compatible with the target GPU.

Expected project layout after copying the package content into MOSS-TTS:

```text
MOSS-TTS/
  models/Indic-Mio/
  vendor/MioTTS-Inference/
  finetuning/runs/indic-mio-spicor-fullshort-stage2/  <- adapter files
  data/voices/simran_clone_indian.mp3
```

## Start the fine-tuned backend

From WSL, inside the recipient's MOSS-TTS project:

```bash
export MIOTTS_MODEL_PATH="models/Indic-Mio"
export MIOTTS_LORA_PATH="finetuning/runs/indic-mio-spicor-fullshort-stage2"
export MIOTTS_LLM_MODEL="indic-mio-spicor-fullshort-stage2"
export MIOTTS_CODEC_MODEL="Aratako/MioCodec-25Hz-44.1kHz-v2"
bash scripts/start-backend.sh
```

In a second terminal:

```bash
source .venv/bin/activate
MIOTTS_MODEL="indic-mio-spicor-fullshort-stage2" \
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

## Production request settings

Always use:

```json
{
  "voice": "simran_clone_indian",
  "language": "English",
  "emotion": "neutral",
  "temperature": 0.55,
  "top_p": 0.87,
  "repetition_penalty": 1.04,
  "max_tokens": 650,
  "seed": 42,
  "cache": true,
  "response_format": "wav"
}
```

Do not add `<happy>` or other emotion tags when voice/accent consistency is more
important than expression. Such tags can change delivery and accent.

## Consistency limitation

The same seed and settings improve repeatability, but this experimental model
can still change accent on short, unseen sentences. For fixed production bot
lines, generate and approve one WAV once, then serve that WAV from cache. That
is the only way to guarantee that the voice does not change between calls.

# Complete integration guide

This guide installs and tests the `indic-mio-spicor-fullshort-stage2` LoRA
adapter with the production Simran Indian-English reference voice.

## 1. Supported setup

The known working setup is:

- Windows 11 with WSL2 Ubuntu, or native Linux
- NVIDIA GPU with approximately 8 GB dedicated VRAM or more
- NVIDIA driver/CUDA access working inside WSL/Linux
- Python 3.12 environments created by the MOSS-TTS setup scripts
- vLLM 0.29.0 in `.venv-vllm`
- The MOSS-TTS gateway environment in `.venv`
- MioTTS-Inference environment under `vendor/MioTTS-Inference/.venv`

The package is a LoRA adapter, not a standalone full model. It needs the
Indic-Mio base model and MioCodec runtime.

## 2. Prepare the recipient's MOSS-TTS project

Set up a clean MOSS-TTS checkout first. The following paths must exist:

```text
MOSS-TTS/models/Indic-Mio/config.json
MOSS-TTS/models/Indic-Mio/model.safetensors
MOSS-TTS/vendor/MioTTS-Inference/run_server.py
MOSS-TTS/.venv-vllm/
MOSS-TTS/.venv/
MOSS-TTS/vendor/MioTTS-Inference/.venv/
```

MioCodec must be identified as:

```text
Aratako/MioCodec-25Hz-44.1kHz-v2
```

Do not pass `models/MioCodec-25Hz-44.1kHz-v2` to this version of
`MioCodecModel.from_pretrained`. It treats that string as a Hugging Face repo
ID and fails. The repository ID downloads the codec once and then uses the
Hugging Face cache on later starts.

## 3. Verify this extracted package

From the extracted package directory:

```bash
python3 verify_package.py
```

Do not continue if it reports an error.

## 4. Install automatically

Run this from the extracted package directory and replace the target path when
needed:

```bash
python3 install_into_moss_tts.py \
  --target /mnt/c/Users/USERNAME/Downloads/MOSS-TTS
```

The installer:

1. Validates the target project and base model.
2. Copies the adapter to
   `finetuning/runs/indic-mio-spicor-fullshort-stage2`.
3. Copies the exact Simran reference to
   `data/voices/simran_clone_indian.mp3`.
4. Merges `simran_clone_indian` into `data/voices.json`.
5. Installs dedicated backend and gateway launchers.
6. Installs the reference gateway implementation and requirements.
7. Backs up any file it replaces under `package-backup/<timestamp>/`.

## 5. Start the backend

In WSL terminal 1:

```bash
cd /mnt/c/Users/USERNAME/Downloads/MOSS-TTS
bash scripts/start-finetuned-simran.sh
```

The first launch can take several minutes. Success ends with output similar to:

```text
vLLM is ready; starting MioCodec API on http://127.0.0.1:8001
Application startup complete.
Uvicorn running on http://127.0.0.1:8001
```

The FlashAttention warning from MioCodec is non-fatal; PyTorch SDPA is used as
the fallback.

## 6. Start the gateway

In WSL terminal 2:

```bash
cd /mnt/c/Users/USERNAME/Downloads/MOSS-TTS
bash scripts/start-finetuned-gateway.sh
```

## 7. Confirm the correct model is active

From Windows PowerShell:

```powershell
Invoke-RestMethod "http://127.0.0.1:8080/health" | ConvertTo-Json
```

Expected result:

```json
{
  "gateway": "ready",
  "backend": "ready",
  "model": "indic-mio-spicor-fullshort-stage2"
}
```

## 8. Generate a real uncached test

Use `cache=false` only while confirming that the adapter can generate audio:

```powershell
$body = @{
  input = "Thank you so much. We will connect with you shortly."
  voice = "simran_clone_indian"
  language = "English"
  emotion = "neutral"
  temperature = 0.55
  top_p = 0.87
  repetition_penalty = 1.04
  max_tokens = 650
  seed = 42
  stream = $false
  response_format = "wav"
  cache = $false
} | ConvertTo-Json

Invoke-WebRequest `
  -Uri "http://127.0.0.1:8080/v1/audio/speech" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes($body)) `
  -OutFile "simran-finetuned-test.wav"
```

The file `simran-finetuned-test.wav` should be a valid WAV and should play in
Windows.

## 9. Production settings

Keep all of these fixed for consistent delivery:

```text
voice:              simran_clone_indian
language:           English
emotion:            neutral
temperature:        0.55
top_p:               0.87
repetition_penalty:  1.04
max_tokens:          650
seed:                42
cache:               true
```

Use the exact packaged reference audio. Do not normalize, trim, resample, or
convert it unless a deliberate new voice version is being created. Avoid style
tags such as `<happy>` when accent consistency is the priority.

The model has no explicit speech-speed setting. Temperature, top-p, and
repetition penalty are not speed controls.

## 10. Guaranteeing unchanged fixed responses

The LoRA and fixed seed improve consistency, but generative TTS may still drift
on short or unseen sentences. To guarantee identical production playback:

1. Generate several candidates with `cache=false`.
2. Select and approve the best WAV.
3. Store that WAV in the approved-phrase cache.
4. Use `cache=true` in production.

For dynamic text, evaluate pronunciation and accent before deployment. No
generative TTS configuration can guarantee identical audio for arbitrary text.

## 11. Integration into a different application

The application should call the gateway endpoint:

```text
POST http://127.0.0.1:8080/v1/audio/speech
Content-Type: application/json
```

Use `config/example-request.json` as the request template. The endpoint returns
WAV bytes. Save them directly or stream the completed WAV to the telephony
layer. Indic-Mio currently produces a complete WAV rather than true incremental
audio frames.

## 12. Common failures

### Codec reports `Repo id must be in the form...`

Use exactly:

```bash
export MIOTTS_CODEC_MODEL="Aratako/MioCodec-25Hz-44.1kHz-v2"
```

### Gateway returns 404 for `/v1/chat/completions`

The vLLM adapter name and gateway model name must both be:

```text
indic-mio-spicor-fullshort-stage2
```

### Backend takes several minutes to start

This is expected when loading from the Windows `/mnt/c` 9P filesystem. Do not
start a second copy. Wait for the health endpoint.

### CUDA out of memory

Close other GPU applications. The launcher defaults to 55% vLLM GPU memory
utilization because MioCodec must also fit on the GPU.

### Short sentences change accent

Use approved WAV caching for fixed lines. For dynamic short text, generate
multiple candidates and select the best result. This remains a known model
limitation.


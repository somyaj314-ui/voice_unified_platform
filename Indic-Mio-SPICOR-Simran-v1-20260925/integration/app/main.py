from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field


from app.hinglish import normalize_hinglish


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
VOICE_DIR = DATA_DIR / "voices"
CACHE_DIR = DATA_DIR / "cache"
VOICE_INDEX = DATA_DIR / "voices.json"
APPROVED_PHRASE_DIR = DATA_DIR / "approved_phrases"
APPROVED_PHRASE_INDEX = DATA_DIR / "approved_phrases.json"
BACKEND_URL = os.getenv("MIOTTS_BACKEND_URL", "http://127.0.0.1:8001").rstrip("/")
# This must match the model ID advertised by the local vLLM /v1/models endpoint.
# start-backend.sh serves the downloaded checkpoint under this local-path ID.
MODEL = os.getenv("MIOTTS_MODEL", "models/Indic-Mio")
TIMEOUT_SECONDS = float(os.getenv("BACKEND_TIMEOUT_SECONDS", "180"))

for directory in (VOICE_DIR, CACHE_DIR, APPROVED_PHRASE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Indic-Mio Hindi-English Gateway", version="2.0.0")


def _phrase_key(text: str, voice: str, language: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    normalized = re.sub(r"[.!?।]+$", "", normalized).strip()
    return f"{voice}|{language.lower()}|{normalized}"


def _approved_phrase_path(request: "SpeechRequest") -> Path | None:
    if not APPROVED_PHRASE_INDEX.exists():
        return None
    try:
        index = json.loads(APPROVED_PHRASE_INDEX.read_text(encoding="utf-8"))
        relative = index.get(_phrase_key(request.input, request.voice, request.language))
        if not relative:
            return None
        path = (ROOT / relative).resolve()
        path.relative_to(APPROVED_PHRASE_DIR.resolve())
        return path if path.is_file() else None
    except (json.JSONDecodeError, OSError, ValueError):
        return None


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1, max_length=4000)
    voice: str
    language: Literal["Hindi", "English", "Hinglish", "Auto"] = "Hinglish"
    emotion: Literal[
        "neutral",
        "active",
        "happy",
        "sad",
        "angry",
        "whisper",
        "enunciated",
        "confused",
        "disgust",
        "fear",
        "surprise",
    ] = "neutral"
    stream: bool = False
    response_format: Literal["wav"] = "wav"
    temperature: float = 0.8
    top_p: float = 1.0
    max_tokens: int = Field(default=700, ge=64, le=2048)
    repetition_penalty: float = 1.0
    seed: int = Field(default=42, ge=0)
    cache: bool = True


def _load_voices() -> dict[str, dict[str, str]]:
    if not VOICE_INDEX.exists():
        return {}
    return json.loads(VOICE_INDEX.read_text(encoding="utf-8"))


def _save_voices(voices: dict[str, dict[str, str]]) -> None:
    VOICE_INDEX.parent.mkdir(parents=True, exist_ok=True)
    temp = VOICE_INDEX.with_suffix(".tmp")
    temp.write_text(json.dumps(voices, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(VOICE_INDEX)


def _voice_audio_path(voice: dict[str, str]) -> Path:
    path = Path(voice["audio_path"])
    return path if path.is_absolute() else ROOT / path


def _normalize_text(text: str, language: str = "Auto") -> str:
    return normalize_hinglish(text, language)


def _shape_delivery(text: str, emotion: str) -> str:
    """Apply Indic-Mio's documented sentence-suffix emotion/style tags."""
    if emotion == "active":
        text = re.sub(r"!{2,}", "!", text).strip()
    elif emotion != "neutral":
        tag = f"<{emotion}>"
        # The official model card requires the tag after the sentence. Avoid
        # duplicating a tag when callers already supplied one themselves.
        text = re.sub(
            rf"([.!?।])(?!\s*{re.escape(tag)})",
            rf"\1 {tag}",
            text,
        ).strip()
        if text and not re.search(r"[.!?।]\s*<[^>]+>\s*$", text):
            text = f"{text}. {tag}"
    return text


def _cache_key(request: SpeechRequest, voice: dict[str, str]) -> str:
    audio_path = _voice_audio_path(voice)
    audio_version = f"{audio_path.stat().st_size}:{audio_path.stat().st_mtime_ns}"
    payload = {
        "model": MODEL,
        "text": _normalize_text(request.input, request.language),
        "voice": request.voice,
        "voice_audio": audio_version,
        "transcript": voice["transcript"],
        "language": request.language,
        "emotion": request.emotion,
        "format": request.response_format,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "repetition_penalty": request.repetition_penalty,
        "seed": request.seed,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _backend_payload(request: SpeechRequest, voice: dict[str, str]) -> dict:
    text = _normalize_text(request.input, request.language)
    text = _shape_delivery(text, request.emotion)
    return {
        "text": text,
        "model": MODEL,
        "temperature": str(request.temperature),
        "top_p": str(request.top_p),
        "max_tokens": str(request.max_tokens),
        "repetition_penalty": str(request.repetition_penalty),
        "seed": str(request.seed),
        "output_format": "wav",
        "best_of_n_enabled": "false",
    }


@app.get("/health")
async def health() -> JSONResponse:
    backend = "unreachable"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{BACKEND_URL}/health")
            backend = "ready" if response.is_success else f"http-{response.status_code}"
    except httpx.HTTPError:
        pass
    return JSONResponse({"gateway": "ready", "backend": backend, "model": MODEL})


@app.get("/v1/voices")
async def list_voices() -> dict:
    voices = _load_voices()
    return {
        "voices": [
            {"voice_id": voice_id, "transcript": item["transcript"]}
            for voice_id, item in sorted(voices.items())
        ]
    }


@app.post("/v1/voices/register")
async def register_voice(
    voice_id: str = Form(..., min_length=2, max_length=64),
    transcript: str = Form(..., min_length=2),
    audio: UploadFile = File(...),
) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", voice_id):
        raise HTTPException(400, "voice_id may contain letters, numbers, _ and - only")
    suffix = Path(audio.filename or "voice.wav").suffix.lower()
    if suffix not in {".wav", ".flac", ".mp3", ".m4a", ".ogg"}:
        raise HTTPException(400, "Supported reference formats: wav, flac, mp3, m4a, ogg")
    target = VOICE_DIR / f"{voice_id}{suffix}"
    content = await audio.read()
    if not 10_000 <= len(content) <= 50_000_000:
        raise HTTPException(400, "Reference audio must be between 10 KB and 50 MB")
    target.write_bytes(content)
    voices = _load_voices()
    voices[voice_id] = {
        "audio_path": str(target.resolve()),
        "transcript": transcript.strip(),
    }
    _save_voices(voices)
    return {"voice_id": voice_id, "audio_path": str(target), "registered": True}


@app.post("/v1/audio/speech")
async def speech(request: SpeechRequest):
    if request.stream:
        raise HTTPException(
            400,
            "Indic-Mio currently returns complete WAV files. Stream sentence-sized requests from the client.",
        )
    voices = _load_voices()
    voice = voices.get(request.voice)
    if not voice:
        raise HTTPException(404, f"Unknown voice '{request.voice}'. Register it first.")

    # Approved phrases are used only when production caching is enabled.
    # Evaluation requests with cache=false always reach the model.
    approved = _approved_phrase_path(request) if request.cache else None
    if approved is not None:
        return FileResponse(approved, media_type="audio/wav", headers={"X-TTS-Cache": "APPROVED"})

    extension = ".wav"
    cache_file = CACHE_DIR / f"{_cache_key(request, voice)}{extension}"
    media_type = "audio/wav"
    if request.cache and cache_file.exists():
        return FileResponse(cache_file, media_type=media_type, headers={"X-TTS-Cache": "HIT"})

    payload = _backend_payload(request, voice)
    audio_path = _voice_audio_path(voice)
    mime_type = mimetypes.guess_type(audio_path.name)[0] or "audio/wav"
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        with audio_path.open("rb") as reference:
            response = await client.post(
                f"{BACKEND_URL}/v1/tts/file",
                data=payload,
                files={"reference_audio": (audio_path.name, reference, mime_type)},
            )
    if not response.is_success:
        raise HTTPException(response.status_code, response.text[:2000])
    audio = response.content
    if not audio.startswith(b"RIFF"):
        raise HTTPException(502, "MioTTS returned invalid WAV data")
    if request.cache:
        cache_file.write_bytes(audio)
    return Response(
        audio,
        media_type=media_type,
        headers={"X-TTS-Cache": "MISS"},
    )

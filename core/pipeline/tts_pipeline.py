"""
tts_pipeline.py v4 — Indic-Mio with chunk streaming
==============================================================================
- synthesize() and generate_audio() — blocking, full audio return (unchanged)
- generate_audio_streaming() — NEW: yields mulaw bytes chunk by chunk
- _split_into_chunks() — extracted helper (shared by both paths)
"""

from pathlib import Path
import os
import requests
import io
import unicodedata
import re
import uuid
import hashlib
import numpy as np
from core.audio_utils import float32_16k_to_mulaw_8k, resample_float32, wav_file_to_float32
from .tts_engine import get_tts, MIO_TTS_URL, _tts_preset_id

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

_DEFAULT_LANG = "en"


def _normalize_lang(language: str | None) -> str:
    if not language:
        return _DEFAULT_LANG
    code = language.strip().lower().split("-")[0]
    return code if code in ["en", "hi"] else _DEFAULT_LANG


def _clean_text(text: str) -> str:
    """Normalize and strip unsupported characters for Mio."""
    cleaned = text.replace("'", "").replace("-", " ")
    return unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode("ascii")


def _split_into_chunks(clean_text: str, max_chars: int = 250) -> list[str]:
    """Split text into Mio-safe chunks by sentence, then by comma, then by word."""
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', clean_text.strip())
    current_chunk: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            # flush current
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
            # split long sentence by comma/semicolon
            parts = re.split(r'(?<=[,;])\s+', sentence)
            for part in parts:
                part = part.strip()
                if len(part) > max_chars:
                    # split by word
                    words = part.split()
                    for word in words:
                        if current_len + len(word) + 1 > max_chars:
                            if current_chunk:
                                chunks.append(" ".join(current_chunk))
                            current_chunk = [word]
                            current_len = len(word)
                        else:
                            current_chunk.append(word)
                            current_len += len(word) + 1
                else:
                    if current_len + len(part) + 1 > max_chars:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        current_chunk = [part]
                        current_len = len(part)
                    else:
                        current_chunk.append(part)
                        current_len += len(part) + 1
        else:
            if current_len + len(sentence) + 1 > max_chars:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_len = len(sentence)
            else:
                current_chunk.append(sentence)
                current_len += len(sentence) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c.strip() for c in chunks if c.strip()]


def _mio_chunk_to_mulaw(chunk_text: str) -> bytes:
    """
    Send one chunk to Mio, return mulaw 8kHz bytes.
    Raises on HTTP error so caller can handle/skip.
    """
    resp = requests.post(MIO_TTS_URL, json={
        "text": chunk_text,
        "reference": {"type": "preset", "preset_id": _tts_preset_id},
        "llm": {
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 700,
        },
        "output": {"format": "wav"},
    }, timeout=30)

    if not resp.ok:
        raise Exception(f"Indic-Mio HTTP {resp.status_code}: {resp.text[:200]}")

    content_type = resp.headers.get("content-type", "")
    if "audio" in content_type:
        audio_bytes = resp.content
    else:
        import base64 as _b64
        data = resp.json()
        audio_bytes = _b64.b64decode(data["audio"])

    wav_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if sr != 16000:
        wav_np = resample_float32(wav_np, sr, 16000)
    return float32_16k_to_mulaw_8k(wav_np)


def synthesize(text, output_path=Path("speech.wav"), language=_DEFAULT_LANG):
    """Blocking synthesis — writes WAV file to output_path."""
    text = text.strip()
    if not text:
        return
    lang = _normalize_lang(language)
    try:
        tts_engine = get_tts()
        mode = tts_engine.get_mode()
        if mode == "mio" and SOUNDFILE_AVAILABLE:
            _synthesize_mio(text, output_path, tts_engine)
        elif mode == "pyttsx3":
            _synthesize_pyttsx3(text, output_path, tts_engine)
    except Exception as e:
        print(f"[TTS] Synthesis failed: {e}")


def _synthesize_mio(text: str, output_path: Path, tts_engine) -> None:
    """Blocking: synthesize all chunks, concatenate, write WAV file."""
    try:
        if not SOUNDFILE_AVAILABLE:
            raise ImportError("soundfile not available")

        clean_text = _clean_text(text)
        chunks = _split_into_chunks(clean_text)
        if not chunks:
            return

        chunk_audios = []
        sr = None

        for idx, chunk in enumerate(chunks):
            resp = requests.post(MIO_TTS_URL, json={
                "text": chunk,
                "reference": {"type": "preset", "preset_id": _tts_preset_id},
                "llm": {
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "max_tokens": 700,
                },
                "output": {"format": "wav"},
            }, timeout=30)

            if not resp.ok:
                raise Exception(f"Indic-Mio HTTP {resp.status_code}")

            content_type = resp.headers.get("content-type", "")
            if "audio" in content_type:
                audio_bytes = resp.content
            else:
                import base64
                data = resp.json()
                audio_bytes = base64.b64decode(data["audio"])

            wav_np, chunk_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            chunk_audios.append(wav_np)
            sr = chunk_sr

        if chunk_audios:
            combined_wav = np.concatenate(chunk_audios)
            sf.write(str(output_path), combined_wav, sr)

    except Exception:
        _synthesize_pyttsx3(text, output_path, tts_engine)


def _synthesize_pyttsx3(text: str, output_path: Path, tts_engine) -> None:
    try:
        if tts_engine.pyttsx3_engine is None:
            raise RuntimeError("pyttsx3 engine not initialized")
        tts_engine.pyttsx3_engine.save_to_file(text, str(output_path))
        tts_engine.pyttsx3_engine.runAndWait()
        print(f"[TTS] pyttsx3 synthesized -> {output_path}")
    except Exception as exc:
        print(f"[TTS] pyttsx3 synthesis failed: {exc}")


class TTSPipeline:
    def __init__(self, temp_dir: str = ".cache/tts_temp", cache_dir: str = ".cache/tts_cache"):
        self.temp_dir = Path(temp_dir)
        self.cache_dir = Path(cache_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._purge_stale_temp_files()

    def _cache_key(self, text: str, language: str) -> str:
        normalized = unicodedata.normalize("NFKD", f"{language.strip().lower()}|{text.strip()}")
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _purge_stale_temp_files(self) -> None:
        count = 0
        for f in self.temp_dir.glob("temp_output_*.wav"):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass
        if count:
            pass

    def generate_audio(self, text: str, language: str) -> bytes:
        """
        Blocking: returns complete mulaw 8kHz bytes after full synthesis.
        Used for short/precomputed responses where streaming isn't needed.
        """
        cache_key = self._cache_key(text, language)
        cache_wav = self.cache_dir / f"{cache_key}.wav"
        if cache_wav.exists():
            try:
                audio_f32, sr = wav_file_to_float32(str(cache_wav))
                if sr != 16000:
                    audio_f32 = resample_float32(audio_f32, sr, 16000)
                return float32_16k_to_mulaw_8k(audio_f32)
            except Exception:
                try:
                    cache_wav.unlink()
                except Exception:
                    pass

        temp_wav = self.temp_dir / f"temp_output_{uuid.uuid4().hex}.wav"
        synthesize(text=text, output_path=temp_wav, language=language)

        if not temp_wav.exists():
            return b""

        try:
            audio_f32, sr = wav_file_to_float32(str(temp_wav))
            if sr != 16000:
                audio_f32 = resample_float32(audio_f32, sr, 16000)
            try:
                cache_wav.parent.mkdir(parents=True, exist_ok=True)
                temp_wav.replace(cache_wav)
            except Exception:
                pass
            return float32_16k_to_mulaw_8k(audio_f32)
        except Exception:
            return b""
        finally:
            if temp_wav.exists():
                try:
                    temp_wav.unlink()
                except Exception:
                    pass

    def generate_audio_streaming(self, text: str, language: str):
        """
        ── STREAMING PATH ──────────────────────────────────────────────────────
        Generator: yields mulaw 8kHz bytes as EACH chunk is synthesized.
        Caller streams chunk to phone immediately — no waiting for full audio.

        Usage:
            for chunk_bytes in tts_pipeline.generate_audio_streaming(text, lang):
                await _enqueue_audio(outbox_queue, chunk_bytes)

        Latency profile:
            Before: wait N×3s → then stream everything
            After:  wait 3s   → stream chunk1, wait 3s → stream chunk2 ...
                    User hears first word after 3s instead of N×3s
        """
        if not SOUNDFILE_AVAILABLE:
            yield self.generate_audio(text, language)
            return

        tts_engine = get_tts()
        if tts_engine.get_mode() != "mio":
            yield self.generate_audio(text, language)
            return

        clean_text = _clean_text(text)
        chunks = _split_into_chunks(clean_text)

        if not chunks:
            return

        import time as _time
        wall_start = _time.monotonic()

        for idx, chunk in enumerate(chunks):
            try:
                mulaw_bytes = _mio_chunk_to_mulaw(chunk)
                yield mulaw_bytes
            except Exception:
                continue


tts_pipeline = TTSPipeline()
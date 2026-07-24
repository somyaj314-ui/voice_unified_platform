"""
tts_engine.py
Indic-Mio voice cloning TTS via local HTTP server (separate venv/process).
Fallback: pyttsx3 when Mio server unavailable.
"""

import os
import io
import threading
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

MIO_TTS_URL = os.getenv("MIO_TTS_URL", "http://localhost:8001/v1/tts")


class TTSEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self.pyttsx3_engine = None
        self._mode = "none"
        self.on_tts_complete = None
        self.speaker_wav = self._resolve_speaker_wav()

        if PYTTSX3_AVAILABLE:
            self._load_pyttsx3()

        if REQUESTS_AVAILABLE and SOUNDFILE_AVAILABLE and self._check_mio_server():
            self._mode = "mio"
        elif self.pyttsx3_engine:
            self._mode = "pyttsx3"

    def _resolve_speaker_wav(self) -> str | None:
        if _tts_speaker_wav:
            return _tts_speaker_wav
        env_path = os.getenv("F5_SPEAKER_WAV", "").strip()
        candidates = [
            env_path,
            os.path.join(BASE_DIR, "speakers", "nikita_ref_mono24k.wav"),
            os.path.join(BASE_DIR, "speakers", "voice_preview_nikita.wav"),
            os.path.join(ROOT_DIR, "voice_preview_nikita.wav"),
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return c

        return None

    def _load_pyttsx3(self):
        try:
            self.pyttsx3_engine = pyttsx3.init()
            self.pyttsx3_engine.setProperty("rate", 160)
            self.pyttsx3_engine.setProperty("volume", 0.95)
            voices = self.pyttsx3_engine.getProperty("voices")
            for voice in voices:
                if any(n in voice.name.lower() for n in ["zira", "hazel", "female"]):
                    self.pyttsx3_engine.setProperty("voice", voice.id)
                    break
        except Exception as e:
            self.pyttsx3_engine = None

    def _check_mio_server(self) -> bool:
        try:
            base_url = MIO_TTS_URL.rsplit("/v1/", 1)[0]
            resp = requests.get(f"{base_url}/v1/presets", timeout=3)
            return resp.status_code == 200
        except Exception as e:
            pass
            return False

    def speak(self, text: str, language: str = "english", print_text: bool = True) -> None:
        if not text or not text.strip():
            return
        try:
            with self._lock:
                if self._mode == "mio":
                    self._speak_mio(text)
                elif self._mode == "pyttsx3":
                    self._speak_pyttsx3(text)
                else:
                    time.sleep(max(1.5, len(text) / 20))
        finally:
            if callable(self.on_tts_complete):
                try:
                    self.on_tts_complete()
                except Exception as e:
                    print(f"[TTS] Callback error: {e}")

    def _speak_mio(self, text: str) -> None:
        try:
            import numpy as np
            import sounddevice as sd
            import re

            # 1. Clean the text (replace unsupported characters)
            cleaned = text.replace("'", "").replace("-", " ")

            # 2. Split into chunks if longer than 250 characters
            chunks = []
            sentences = re.split(r'(?<=[.!?])\s+', cleaned.strip())
            current_chunk = []
            current_len = 0
            max_chars = 250
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(sentence) > max_chars:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_len = 0
                    parts = re.split(r'(?<=[,;])\s+', sentence)
                    for part in parts:
                        part = part.strip()
                        if len(part) > max_chars:
                            words = part.split(" ")
                            temp = []
                            temp_len = 0
                            for word in words:
                                if temp_len + len(word) + 1 > max_chars:
                                    if temp:
                                        chunks.append(" ".join(temp))
                                    temp = [word]
                                    temp_len = len(word)
                                else:
                                    temp.append(word)
                                    temp_len += len(word) + 1
                            if temp:
                                current_chunk = temp
                                current_len = temp_len
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
                
            chunks = [c.strip() for c in chunks if c.strip()]
            if not chunks:
                return

            for idx, chunk in enumerate(chunks):
                resp = requests.post(
                    MIO_TTS_URL,
                    json={
                        "text": chunk,
                        "reference": {"type": "preset", "preset_id": _tts_preset_id},
                        "output": {"format": "wav"}
                    },
                    timeout=30,
                )
                
                if not resp.ok:
                    raise Exception(f"Indic-Mio HTTP {resp.status_code}")

                wav_np, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
                sd.play(wav_np, samplerate=sr, blocking=True)

        except Exception as e:
            self._speak_pyttsx3(text)

    def _speak_pyttsx3(self, text: str):
        try:
            self.pyttsx3_engine.say(text)
            self.pyttsx3_engine.runAndWait()
        except Exception as e:
            print(f"[TTS] pyttsx3 error: {e}")

    def synthesize(self, text: str, language: str = "en") -> bytes:
        """Return mulaw 8kHz bytes for telephony streaming.

        Used by server.py for parallel TTS generation (e.g. name-confirm
        audio during prerecorded intro playback).  Unlike speak(), this
        does NOT play to speakers — it returns raw bytes.
        """
        if not text or not text.strip():
            return b""
        from pipeline.tts_pipeline import tts_pipeline
        return tts_pipeline.generate_audio(text=text, language=language)

    def get_mode(self) -> str:
        return self._mode

    def get_conditioning_latents(self):
        """Indic-Mio (HTTP) does not expose conditioning latents."""
        return None, None


_tts = None
_tts_speaker_wav: str | None = None
_tts_preset_id: str = "my_speaker_saavi"

def configure_tts(speaker_wav: str | None = None, mio_url: str | None = None, preset_id: str | None = None) -> None:
    global _tts_speaker_wav, _tts_preset_id, MIO_TTS_URL
    if speaker_wav:
        _tts_speaker_wav = speaker_wav
    if mio_url:
        MIO_TTS_URL = mio_url
    if preset_id:
        _tts_preset_id = preset_id

def get_tts() -> TTSEngine:
    global _tts
    if _tts is None:
        _tts = TTSEngine()
    return _tts
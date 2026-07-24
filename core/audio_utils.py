
import audioop
import io
import struct
import wave
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from math import gcd
from collections import deque
import time

# ─────────────────────────────────────────────
# mulaw ↔ PCM
# ─────────────────────────────────────────────

def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """Decode µ-law encoded bytes to signed 16-bit PCM bytes."""
    return audioop.ulaw2lin(mulaw_bytes, 2)  # 2 = 2 bytes per sample (16-bit)


def pcm16_to_mulaw(pcm16_bytes: bytes) -> bytes:
    """Encode signed 16-bit PCM bytes to µ-law bytes."""
    return audioop.lin2ulaw(pcm16_bytes, 2)


# ─────────────────────────────────────────────
# Resampling
# ─────────────────────────────────────────────

def resample_pcm16(pcm16_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample PCM16 audio from one sample rate to another."""
    if from_rate == to_rate:
        return pcm16_bytes
    resampled, _ = audioop.ratecv(pcm16_bytes, 2, 1, from_rate, to_rate, None)
    return resampled


def resample_float32(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample float32 numpy array from from_rate to to_rate."""
    if from_rate == to_rate:
        return audio
    divisor = gcd(from_rate, to_rate)
    up = to_rate // divisor
    down = from_rate // divisor
    return resample_poly(audio, up, down).astype(np.float32)


# ─────────────────────────────────────────────
# Format conversions
# ─────────────────────────────────────────────

def pcm16_bytes_to_float32(pcm16_bytes: bytes) -> np.ndarray:
    """Convert raw PCM16 bytes to float32 numpy array normalised to [-1, 1]."""
    samples = np.frombuffer(pcm16_bytes, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 numpy array [-1, 1] to raw PCM16 bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


# ─────────────────────────────────────────────
# Full pipeline helpers
# ─────────────────────────────────────────────

def mulaw_8k_to_float32_16k(mulaw_bytes: bytes) -> np.ndarray:
    """
    Convert µ-law 8kHz bytes (from Go2Market/FreeSWITCH) to float32 16kHz numpy
    array ready for Whisper ingestion.
    """
    pcm16 = mulaw_to_pcm16(mulaw_bytes)
    pcm16_16k = resample_pcm16(pcm16, 8000, 16000)
    return pcm16_bytes_to_float32(pcm16_16k)


def float32_16k_to_mulaw_8k(audio: np.ndarray) -> bytes:
    """
    Convert float32 16kHz numpy array (TTS output) to µ-law 8kHz bytes
    ready to send back over Go2Market/FreeSWITCH WebSocket.
    """
    audio_8k = resample_float32(audio, 16000, 8000)
    pcm16 = float32_to_pcm16_bytes(audio_8k)
    return pcm16_to_mulaw(pcm16)


# ─────────────────────────────────────────────
# WAV helpers
# ─────────────────────────────────────────────

def wav_file_to_float32(path: str) -> tuple[np.ndarray, int]:
    """Read a WAV file and return (float32 array, sample_rate)."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # stereo → mono
    return audio, sr


def float32_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to in-memory WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def wav_bytes_to_float32(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Parse WAV bytes and return (float32 array, sample_rate)."""
    buf = io.BytesIO(wav_bytes)
    audio, sr = sf.read(buf, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


# ─────────────────────────────────────────────
# Voice Activity Detection (energy-based helper)
# ─────────────────────────────────────────────

def compute_rms(audio: np.ndarray) -> float:
    """Compute RMS energy of an audio chunk."""
    return float(np.sqrt(np.mean(audio ** 2)))


def suppress_echo_like_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    attenuation: float = 0.8,
) -> np.ndarray:
    """Attenuate sustained, low-variance audio that often looks like echo leakage.

    This is intentionally lightweight and conservative: it only reduces gain when
    the signal is very flat across short windows and has moderate RMS, which is a
    common signature of telephony echo or playback bleed.
    """
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.size == 0:
        return audio

    peak = float(np.max(np.abs(audio)))
    if not np.isfinite(peak) or peak <= 1e-6:
        return audio

    normalized = audio / peak
    window_samples = max(1, int(0.02 * sample_rate))
    if len(normalized) < window_samples * 2:
        return np.clip(normalized * attenuation * peak, -1.0, 1.0).astype(np.float32)

    windows = [
        normalized[i:i + window_samples]
        for i in range(0, len(normalized) - window_samples + 1, window_samples)
    ]
    if len(windows) < 2:
        return np.clip(normalized * attenuation * peak, -1.0, 1.0).astype(np.float32)

    window_rms = np.array([compute_rms(w) for w in windows], dtype=np.float32)
    avg_rms = float(np.mean(window_rms))
    rms_variation = float(np.std(window_rms))
    repeated_samples = float(np.mean(np.abs(np.diff(normalized)) < 0.002))

    should_suppress = (
        avg_rms > 0.015 and rms_variation < 0.004 and repeated_samples > 0.65
    ) or (
        avg_rms > 0.03 and rms_variation < 0.008
    )

    if should_suppress:
        normalized = normalized * attenuation

    return np.clip(normalized * peak, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────
# Silero VAD (Neural — primary speech detector)
# ─────────────────────────────────────────────

# Silero VAD requires EXACTLY 512 samples per inference call at 16kHz
# (256 samples at 8kHz). Passing the wrong size silently gives bad probs.
_SILERO_CHUNK_SIZE_16K = 512   # 32ms at 16kHz
_SILERO_CHUNK_SIZE_8K  = 256   # 32ms at 8kHz

SILERO_AVAILABLE = False
_silero_load_error: str | None = None
_silero_repo_path: str | None = None
_silero_repo_lookup_done = False


def _find_local_silero_repo() -> str | None:
    """Resolve the checked-in Silero repo once so each VAD can load its own model."""
    global _silero_repo_path, _silero_repo_lookup_done

    if _silero_repo_lookup_done:
        return _silero_repo_path

    _silero_repo_lookup_done = True
    try:
        import os

        project_root = os.getcwd()
        for entry in os.listdir(project_root):
            if not entry.startswith("snakers4-silero-vad") or not os.path.isdir(entry):
                continue

            nested = os.path.join(entry, "snakers4-silero-vad-980b17e")
            if os.path.isdir(nested) and os.path.exists(os.path.join(nested, "hubconf.py")):
                _silero_repo_path = nested
                break
            if os.path.exists(os.path.join(entry, "hubconf.py")):
                _silero_repo_path = entry
                break
    except Exception:
        _silero_repo_path = None

    return _silero_repo_path


def _load_silero_model(device: str = "cpu"):
    """
    Load a fresh Silero VAD model for one caller/consumer instance.
    The model is stateful, so sharing one module across the flow engine and
    segmenter causes hidden-state bleed between independent audio streams.
    """
    global SILERO_AVAILABLE, _silero_load_error

    local_repo_path = _find_local_silero_repo()

    try:
        if local_repo_path:
            print(f"[VAD] Attempting local Hub load from: {local_repo_path}")
            try:
                model, _ = torch.hub.load(
                    repo_or_dir=local_repo_path,
                    model="silero_vad",
                    source="local",
                    trust_repo=True,
                )
            except Exception as hub_exc:
                print(f"[VAD] Hub load failed ({hub_exc}), trying direct JIT load...")
                # Fallback: Try to find the .jit model file directly in the local repo
                # In Silero V6, it's usually in src/silero_vad/data/silero_vad.jit
                jit_path = os.path.join(local_repo_path, "src", "silero_vad", "data", "silero_vad.jit")
                if not os.path.exists(jit_path):
                    # Try other common locations
                    jit_path = os.path.join(local_repo_path, "silero_vad.jit")
                
                if os.path.exists(jit_path):
                    print(f"[VAD] Found JIT model at: {jit_path}")
                    model = torch.jit.load(jit_path, map_location=device)
                else:
                    raise hub_exc # Re-raise if we can't find a fallback
        else:
            print(f"[VAD] Local Silero not found. Attempting network load...")
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
        
        model.to(device)
        model.eval()
        SILERO_AVAILABLE = True
        print(f"[VAD] Silero VAD loaded OK (device={device})")
        return model
    except Exception as exc:
        _silero_load_error = str(exc)
        SILERO_AVAILABLE = False
        print(f"[VAD] Silero VAD failed to load: {exc}. RMS fallback will be used.")
        return None


class SileroVAD:
    """
    Thin wrapper around an instance-owned Silero VAD model.

    Critical: Silero requires exactly 512 samples at 16kHz per inference call.
    Frames of any other size are zero-padded or truncated before passing to
    the model — never trust Silero on a wrong-sized tensor.
    """

    def __init__(self, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._model = _load_silero_model(device)

    @property
    def available(self) -> bool:
        return self._model is not None

    def is_speech(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        threshold: float = 0.5,
    ) -> bool:
        """
        Returns True if any part of the audio frame contains speech.
        Processes audio in steps of Silero's required chunk size.
        """
        return self.speech_probability(audio, sr) > threshold

    def speech_probability(
        self,
        audio: np.ndarray,
        sr: int = 16000,
    ) -> float:
        """
        Return the maximum speech probability found in the audio chunks.
        Processes audio in Silero's required 512-sample steps (at 16kHz).
        """
        if not self.available:
            return 0.0
        if audio is None or (hasattr(audio, 'size') and audio.size == 0):
            return 0.0

        required = _SILERO_CHUNK_SIZE_16K if sr >= 16000 else _SILERO_CHUNK_SIZE_8K
        audio_flat = np.asarray(audio, dtype=np.float32).flatten()
        
        # If audio is empty after flattening
        if audio_flat.size == 0:
            return 0.0

        max_prob = 0.0
        # Process in chunks of `required` samples
        for i in range(0, len(audio_flat), required):
            chunk = audio_flat[i : i + required]
            if len(chunk) < required:
                # Pad last chunk if needed
                chunk = np.pad(chunk, (0, required - len(chunk)))
            
            try:
                tensor = torch.from_numpy(chunk).to(self.device)
                with torch.no_grad():
                    prob = self._model(tensor, sr).item()
                max_prob = max(max_prob, float(prob))
            except Exception as exc:
                print(f"[VAD] Silero inference error in chunk {i}: {exc}")
                continue

        return float(np.clip(max_prob, 0.0, 1.0))


    def reset_state(self):
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()
# ─────────────────────────────────────────────
# VADAccumulator (simple energy-based, unchanged)
# ─────────────────────────────────────────────

class VADAccumulator:
    """
    Accumulates audio chunks and detects utterance boundaries using RMS energy.

    Usage:
        vad = VADAccumulator()
        for chunk in stream:
            audio = mulaw_8k_to_float32_16k(chunk)
            if vad.push(audio):
                utterance = vad.get_utterance()
                vad.reset()
                # process utterance
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_threshold: float = 0.01,
        min_speech_duration_s: float = 0.3,
        silence_duration_s: float = 0.8,
    ):
        self.sr = sample_rate
        self.threshold = silence_threshold
        self.min_speech_samples = int(min_speech_duration_s * sample_rate)
        self.silence_samples = int(silence_duration_s * sample_rate)

        self._buffer: list[np.ndarray] = []
        self._total_samples = 0
        self._speech_started = False
        self._silence_counter = 0

    def push(self, chunk: np.ndarray) -> bool:
        """
        Push an audio chunk. Returns True when an utterance is complete
        (speech followed by enough silence).
        """
        self._buffer.append(chunk)
        self._total_samples += len(chunk)

        rms = compute_rms(chunk)
        if rms >= self.threshold:
            self._speech_started = True
            self._silence_counter = 0
        else:
            if self._speech_started:
                self._silence_counter += len(chunk)

        if (
            self._speech_started
            and self._silence_counter >= self.silence_samples
            and self._total_samples >= self.min_speech_samples
        ):
            return True
        return False

    def get_utterance(self) -> np.ndarray:
        """Return the accumulated audio as a single float32 array."""
        return np.concatenate(self._buffer) if self._buffer else np.array([], dtype=np.float32)

    def reset(self) -> None:
        """Reset the accumulator for the next utterance."""
        self._buffer = []
        self._total_samples = 0
        self._speech_started = False
        self._silence_counter = 0

    @property
    def has_speech(self) -> bool:
        return self._speech_started


# ─────────────────────────────────────────────
# StreamingSpeechSegmenter (Silero primary, RMS fallback)
# ─────────────────────────────────────────────

class StreamingSpeechSegmenter:

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 32,
        min_speech_duration_s: float = 0.2,    # allow short telephony acknowledgements
        max_chunk_duration_s: float = 15.0,    # v3.4: was 2.0
        max_silence_duration_s: float = 2.5,   # v3.4: was 0.8
        short_pause_emit_ms: int = 0,         # v3.4: was 100
        preroll_ms: int = 160,
        target_rms: float = 0.08,
        max_gain: float = 3.0,                 # v3.4: was 6.0
        min_frame_rms: float = 0.05,
        speech_ratio: float = 3.0,
        use_silero: bool = True,                # v3.5: Silero as primary VAD
        silero_threshold: float = 0.5,          # v3.5: speech probability cutoff
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)

        # Keep a low floor so short acknowledgements still reach ASR.
        computed_min = int(min_speech_duration_s * sample_rate / self.frame_samples)
        self.min_speech_frames = max(6, computed_min)

        self.max_chunk_frames = max(1, int(max_chunk_duration_s * sample_rate / self.frame_samples))
        self.max_silence_frames = max(1, int(max_silence_duration_s * sample_rate / self.frame_samples))
        self.short_pause_emit_frames = max(1, int(short_pause_emit_ms / frame_ms))
        self.target_rms = target_rms
        self.max_gain = max_gain
        self.min_frame_rms = min_frame_rms
        self.speech_ratio = speech_ratio
        self.noise_floor = min_frame_rms
        self.silero_threshold = silero_threshold

        # ── Silero setup ──────────────────────────────────────────────────────
        # _load_silero_model() handles all errors internally; no crash risk here.
        self.use_silero = use_silero
        self.silero: SileroVAD | None = None
        if use_silero:
            self.silero = SileroVAD()

        self.preroll = deque(maxlen=max(1, preroll_ms // frame_ms))
        self.speech_frames: list[np.ndarray] = []
        self.speech_frame_count = 0
        self.trailing_silence = 0
        self.partial = np.array([], dtype=np.float32)

        vad_mode = (
            "Silero neural (primary)"
            if (use_silero and self.silero is not None and self.silero.available)
            else "RMS energy (Silero load failed — fallback)"
            if use_silero
            else "RMS energy"
        )
        self.ignore_until: float = 0.0
        self.tts_active: bool = False
        print(
            f"[VAD] StreamingSpeechSegmenter v3.5 — "
            f"vad={vad_mode}, "
            f"min_speech_frames={self.min_speech_frames} "
            f"({self.min_speech_frames * frame_ms}ms), "
            f"short_pause_emit={short_pause_emit_ms}ms, "
            f"max_silence={max_silence_duration_s}s, "
            f"max_chunk={max_chunk_duration_s}s, "
            f"max_gain={max_gain}"
        )

    # ── public interface ──────────────────────────────────────────────────────

    def push(self, chunk: np.ndarray) -> list[np.ndarray]:
        """Consume a streaming chunk and return zero or more speech-only segments."""

        # Block VAD entirely while TTS is active or during cooldown window
        if self.tts_active or time.time() < self.ignore_until:
            return []
        normalized = self._normalize_gain(chunk)
        if normalized.size == 0:
            return []

        merged = np.concatenate((self.partial, normalized))
        frame_count = len(merged) // self.frame_samples
        if frame_count == 0:
            self.partial = merged
            return []

        segments: list[np.ndarray] = []
        limit = frame_count * self.frame_samples
        for start in range(0, limit, self.frame_samples):
            frame = merged[start: start + self.frame_samples]
            segments.extend(self._consume_frame(frame))

        self.partial = merged[limit:]
        return segments

    def flush(self) -> list[np.ndarray]:
        """Emit any buffered speech when the stream ends."""
        segments: list[np.ndarray] = []
        if self.partial.size:
            padded = np.pad(self.partial, (0, max(0, self.frame_samples - len(self.partial))))
            segments.extend(self._consume_frame(padded[: self.frame_samples]))
            self.partial = np.array([], dtype=np.float32)
        final = self._finalize_segment(trim_trailing_silence=True)
        if final is not None:
            segments.append(final)
        return segments

    def is_quality_audio(self, audio: np.ndarray, min_duration_s: float = 0.6) -> bool:
        if audio.size == 0:
            return False

        duration = len(audio) / self.sample_rate
        if duration < min_duration_s:
            if compute_rms(audio) > 0.015:
                return True  # short but strong — likely a quick word
            print(f"[VAD] Rejected segment: too short ({duration:.2f}s < {min_duration_s}s)")
            return False

        rms = compute_rms(audio)
        if rms < 0.01:
            print(f"[VAD] Rejected segment: too quiet (rms={rms:.4f})")
            return False

        # Spectral variation: real speech has changing RMS over 50ms windows
        try:
            window_size = int(0.05 * self.sample_rate)
            if len(audio) < window_size * 2:
                return True

            windows = [
                audio[i: i + window_size]
                for i in range(0, len(audio) - window_size, window_size)
            ]
            if len(windows) < 2:
                return True

            rms_values = [compute_rms(w) for w in windows]
            rms_variation = float(np.std(rms_values))

            if rms_variation < 0.002:
                print(f"[VAD] Rejected segment: flat signal (variation={rms_variation:.4f})")
                return False
            if rms_variation > 0.5:
                print(f"[VAD] Rejected segment: noisy signal (variation={rms_variation:.4f})")
                return False

            return True
        except Exception:
            return True

    # ── private helpers ───────────────────────────────────────────────────────


    def _normalize_gain(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32).flatten()
        if chunk.size == 0:
            return chunk
        rms = compute_rms(chunk)
        if rms <= 1e-6:
            return chunk
        # Only amplify if frame has real signal — never boost silence
        if rms > self.min_frame_rms:
            gain = min(self.max_gain, self.target_rms / rms)
        else:
            gain = 1.0
        amplified = chunk * gain
        threshold = 0.9
        mask = np.abs(amplified) > threshold
        if np.any(mask):
            amplified = amplified.copy()
            amplified[mask] = np.sign(amplified[mask]) * (
                threshold
                + (1.0 - threshold)
                * np.tanh((np.abs(amplified[mask]) - threshold) / (1.0 - threshold))
            )
        return amplified.astype(np.float32)
    def set_silence_window(self, seconds: float) -> None:
        """
        Suppress VAD for `seconds` after TTS finishes.
        Call this immediately after TTS audio starts playing.
        """
        import time
        self.ignore_until = time.time() + seconds
        print(f"[VAD] Silence window set for {seconds}s")

    def set_tts_active(self, active: bool) -> None:
        """
        Call set_tts_active(True) when bot starts speaking.
        Call set_tts_active(False) when bot finishes speaking.
        """
        self.tts_active = active
        if active:
            import time
            self.ignore_until = time.time() + 1.2
    def _silero_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Silero requires exactly 512 samples at 16kHz.
        Pad or truncate to enforce this before every inference call.
        """
        required = 512
        if len(frame) > required:
            return frame[:required].astype(np.float32)
        if len(frame) < required:
            return np.pad(frame, (0, required - len(frame))).astype(np.float32)
        return frame.astype(np.float32)
    def _consume_frame(self, frame: np.ndarray) -> list[np.ndarray]:
        frame = np.asarray(frame, dtype=np.float32)
        segments: list[np.ndarray] = []
        rms = compute_rms(frame)

        if self.silero is not None and self.silero.available:
            silero_frame = self._silero_frame(frame)
            is_speech = self.silero.is_speech(silero_frame, sr=self.sample_rate, threshold=self.silero_threshold)
        else:
            is_speech = rms >= self.min_frame_rms

        if is_speech:
            if not self.speech_frames:
                self.speech_frames.extend(self.preroll)
                self.preroll.clear()
            self.speech_frames.append(frame)
            self.speech_frame_count += 1
            self.trailing_silence = 0
        else:
            # Keep adaptive noise floor updated regardless of VAD mode
            # (used as RMS fallback threshold and for diagnostics)
            self.noise_floor = (0.95 * self.noise_floor) + (0.05 * max(rms, 1e-6))
            if self.speech_frames:
                self.trailing_silence += 1
            else:
                self.preroll.append(frame)

        # Condition 1: force-emit when max chunk size reached
        if self.speech_frames and self.speech_frame_count >= self.max_chunk_frames:
            finalized = self._finalize_segment(trim_trailing_silence=True)
            if finalized is not None:
                segments.append(finalized)

        # Condition 2: silence-ended utterance — only when stable
        elif self.speech_frames and self._is_stable_utterance():
            finalized = self._finalize_segment(trim_trailing_silence=True)
            if finalized is not None:
                segments.append(finalized)

        # Condition 3: short-pause adaptive emit
        # Respect the configured minimum speech window, not a hardcoded frame count

        if not self.speech_frames and not is_speech:
            self.preroll.append(frame)

        return segments
    def _is_stable_utterance(self) -> bool:
        """
        True when speech is long enough AND silence has held long enough
        to confirm the utterance is genuinely complete.
        """
        return (
            self.speech_frame_count >= self.min_speech_frames
            and self.trailing_silence >= int(0.8 * self.max_silence_frames)
        )
    def _finalize_segment(self, trim_trailing_silence: bool) -> np.ndarray | None:
        # Drop incomplete segments that never reached the configured speech minimum.
        if not self.speech_frames or self.speech_frame_count < self.min_speech_frames:
            self._reset_segment_state()
            return None

        frames = self.speech_frames
        if trim_trailing_silence and self.trailing_silence:
            keep = max(0, len(frames) - self.trailing_silence)
            frames = frames[:keep]

        if not frames:
            self._reset_segment_state()
            return None

        audio = np.concatenate(frames).astype(np.float32)
        trimmed = self._trim_with_torchaudio(audio)
        result = trimmed if trimmed.size else audio
        self._reset_segment_state()
        return result

    def _reset_segment_state(self) -> None:
        self.speech_frames = []
        self.speech_frame_count = 0
        self.trailing_silence = 0

    def _trim_audio_energy(self, audio: np.ndarray, threshold: float = 0.0015) -> np.ndarray:
        """
        Shave off leading/trailing silence using a simple energy threshold.
        Replaces torchaudio.functional.vad to avoid C++ extension issues on Windows.
        """
        if audio.size == 0:
            return audio

        abs_audio = np.abs(audio)
        mask = abs_audio > threshold

        if not np.any(mask):
            return np.array([], dtype=np.float32)

        start = np.argmax(mask)
        end = len(audio) - np.argmax(mask[::-1])

        # Add 100ms padding
        padding = int(0.1 * self.sample_rate)
        start = max(0, start - padding)
        end = min(len(audio), end + padding)

        return audio[start:end]

    def _trim_with_torchaudio(self, audio: np.ndarray) -> np.ndarray:
        # Uses numpy energy trim (avoids torchaudio C++ issues on Windows)
        return self._trim_audio_energy(audio)

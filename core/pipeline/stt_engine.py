"""
stt_engine.py
Speech-to-Text using OpenAI Whisper Medium via faster-whisper.
Records from microphone and returns transcript.

CHANGELOG v1.2 (Real-time streaming fix):
- transcribe(): Added MIN_AUDIO_DURATION_S = 1.0s gate before calling Whisper.
  Segments shorter than 1 second are returned as "" to prevent hallucinations.
- transcribe(): Added WHISPER_HALLUCINATIONS filter — a set of known phrases
  Whisper produces on silence/noise ("in the", "you", "thank you", etc.).
- transcribe(): Added pre-transcription RMS quality gate (< 0.008 → skip).
- listen(): Added audio length check (< SAMPLE_RATE * 0.5 → retry prompt).
"""

import io
import os
import wave
import tempfile
import sys
import numpy as np
import torch



try:
    import ctranslate2
    CTRANSLATE2_AVAILABLE = True
except ImportError:
    CTRANSLATE2_AVAILABLE = False

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("[STT] sounddevice not available. Mic input will be disabled.")

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    WHISPER_AVAILABLE = False
    WhisperModel = None
    print("[STT] faster-whisper not available. Will use text input fallback.")


SAMPLE_RATE = 8000
CHANNELS = 1
DTYPE = "int16"

# Minimum audio duration before attempting transcription.
# Whisper hallucinates badly on audio < 1.0s — "in the", "you", "thank you", etc.
MIN_AUDIO_DURATION_S = 1.0
MIN_SHORT_AUDIO_DURATION_S = 0.25
MIN_SHORT_AUDIO_RMS = 0.006

# Minimum pre-gain RMS energy to bother calling Whisper.
MIN_PRE_TRANSCRIBE_RMS = 0.008

LANGUAGE_CODE_MAP = {
    "english": "en",
    "en": "en",
}

# Known Whisper hallucination phrases on silence / very short audio.
# Exact lowercase match after stripping punctuation and extra whitespace.
WHISPER_HALLUCINATIONS: set[str] = {
    # English filler / common hallucinations
    "in the",
    "you",
    "thank you",
    "thanks",
    "thank you.",
    "thanks.",
    "you.",
    "hmm",
    "hmm.",
    "hm",
    "hm.",
    "um",
    "uh",
    "ah",
    "oh",
    "",
    "...",
    ".",
    "the",
    "a",
    "and",
    "i",
    "so",
    "well",
    "yeah",
    # Subtitles / credits artifact
    "subtitles by",
    "subtitles",
    "transcript",
    "www.",
    "Thank you. Bye."
    "Bye."
}

# Silence detection thresholds
SILENCE_THRESHOLD = 500      # RMS below this = silence
SILENCE_DURATION_SEC = 1.8   # seconds of silence to stop recording
MAX_RECORD_SEC = 30          # maximum recording duration


def _is_hallucination(text: str) -> bool:
    """
    Return True if `text` looks like a Whisper hallucination on silence/noise.
    Normalises to lowercase, strips leading/trailing punctuation and whitespace.
    """
    if not text:
        return True
    normalized = text.strip().lower().rstrip(".,!?…").strip()
    return normalized in WHISPER_HALLUCINATIONS


class WhisperSTT:
   

    def __init__(self, model_size: str = "Systran/faster-whisper-medium"):
        self.model = None
        self.model_size = model_size

        requested_device = os.getenv("STT_DEVICE", "auto").strip().lower()
        requested_compute_type = os.getenv("STT_COMPUTE_TYPE", "").strip().lower()
        torch_cuda_available = torch.cuda.is_available()
        torch_cuda_version = torch.version.cuda
        ct2_cuda_device_count = 0
        ct2_cuda_compute_types = set()
        ct2_cuda_available = False
        ct2_probe_error = None

        if CTRANSLATE2_AVAILABLE:
            try:
                ct2_cuda_device_count = ctranslate2.get_cuda_device_count()
                ct2_cuda_compute_types = ctranslate2.get_supported_compute_types("cuda")
                ct2_cuda_available = ct2_cuda_device_count > 0
            except Exception as e:
                ct2_probe_error = e

        if requested_device not in {"auto", "cuda", "cpu"}:
            print(f"[STT] Invalid STT_DEVICE='{requested_device}'. Falling back to auto.")
            requested_device = "auto"

        if requested_device == "cuda":
            if ct2_cuda_available:
                self.device = "cuda"
            else:
                print("[STT] STT_DEVICE=cuda requested, but faster-whisper CUDA is not available. Falling back to CPU.")
                self.device = "cpu"
        elif requested_device == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda" if ct2_cuda_available else "cpu"

        default_compute_type = "float16" if self.device == "cuda" else "int8"
        self.compute_type = requested_compute_type or default_compute_type

        print(f"[STT] Python: {sys.executable}")
        print(f"[STT] torch: {torch.__version__}")
        print(f"[STT] torch.version.cuda: {torch_cuda_version}")
        print(f"[STT] torch.cuda.is_available(): {torch_cuda_available}")
        if torch_cuda_version is None:
            print("[STT] Torch in this environment is a CPU-only build.")
        if CTRANSLATE2_AVAILABLE:
            print(f"[STT] ctranslate2: {ctranslate2.__version__}")
            print(f"[STT] ctranslate2 CUDA devices: {ct2_cuda_device_count}")
            if ct2_cuda_compute_types:
                print(f"[STT] ctranslate2 CUDA compute types: {sorted(ct2_cuda_compute_types)}")
            if ct2_probe_error:
                print(f"[STT] ctranslate2 CUDA probe failed: {ct2_probe_error}")
        else:
            print("[STT] ctranslate2 not available.")
        if self.device == "cuda" and not torch_cuda_available:
            print("[STT] Torch is CPU-only, but faster-whisper will use CUDA via CTranslate2.")
        if torch_cuda_available:
            print(f"[STT] CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"[STT] Using device: {self.device} (compute_type={self.compute_type})")
        print(f"[STT] Min audio duration before transcribe: {MIN_AUDIO_DURATION_S}s")
        if WHISPER_AVAILABLE:
            self._load_model()

    def _load_model(self):
        print(f"[STT] Loading Whisper {self.model_size} model (device={self.device})...")
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            print("[STT] Whisper model loaded OK")
        except Exception as e:
            print(f"[STT] Failed to load Whisper model: {e}")
            self.model = None

    def record_audio(self, prompt: str = "🎤 Listening...") -> np.ndarray | None:
        """
        Record audio from microphone until silence is detected.
        Returns numpy array of int16 audio samples.
        """
        if not SOUNDDEVICE_AVAILABLE:
            return None

        print(f"\n{prompt}")

        audio_chunks = []
        silence_chunks = 0
        chunk_size = int(SAMPLE_RATE * 0.1)  # 100ms chunks
        silence_limit = int(SILENCE_DURATION_SEC / 0.1)
        max_chunks = int(MAX_RECORD_SEC / 0.1)
        started_speaking = False

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=chunk_size
            ) as stream:
                chunk_count = 0
                while chunk_count < max_chunks:
                    chunk, _ = stream.read(chunk_size)
                    rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                    audio_chunks.append(chunk.copy())
                    chunk_count += 1

                    if rms > SILENCE_THRESHOLD:
                        started_speaking = True
                        silence_chunks = 0
                    else:
                        if started_speaking:
                            silence_chunks += 1
                            if silence_chunks >= silence_limit:
                                break

        except Exception as e:
            print(f"[STT] Recording error: {e}")
            return None

        if not audio_chunks:
            return None

        audio = np.concatenate(audio_chunks, axis=0).flatten()
        return audio

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        """
        Transcribe numpy audio array (float32, 16kHz) to text using Whisper.
        Uses advanced per-use-case dynamic language detection.
        """
        if self.model is None:
            return ""
        if audio is None or len(audio) == 0:
            return ""

        from core.loader import load_use_case
        import math
        _cfg = load_use_case()
        stt_cfg = _cfg.raw.get("stt", {})
        
        # Load configs
        acoustic_probe_enabled = stt_cfg.get("acoustic_probe_enabled", False)
        mismatch_margin = float(stt_cfg.get("mismatch_margin", 0.20))
        short_clip_threshold_sec = float(stt_cfg.get("short_clip_threshold_sec", 3.0))
        conf_gates = stt_cfg.get("confidence_gate", {"en": 0.45, "hi": 0.50})
        
        expected_lang = str(language).strip().lower()
        if expected_lang not in {"en", "hi", "ur"}:
            expected_lang = "en"
            
        whisper_lang = LANGUAGE_CODE_MAP.get(expected_lang, "en")

        audio_sr = 16000
        duration_s = len(audio) / audio_sr
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

        if duration_s < MIN_SHORT_AUDIO_DURATION_S:
            return ""
        if duration_s < MIN_AUDIO_DURATION_S and rms < MIN_SHORT_AUDIO_RMS:
            return ""
        if rms < MIN_PRE_TRANSCRIBE_RMS:
            return ""

        print(f"[STT] Transcribing: dur={duration_s:.2f}s rms={rms:.4f}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(audio_sr)
                pcm16 = np.clip(audio, -1.0, 1.0)
                pcm16 = (pcm16 * 32767).astype(np.int16)
                wf.writeframes(pcm16.tobytes())

            mismatch_detected = False
            
            # Stage 1 & 2: Acoustic Probe
            if acoustic_probe_enabled:
                try:
                    # Faster-whisper model.detect_language expects a numpy array or file object.
                    lang, prob, all_probs = self.model.detect_language(audio)
                    probs_dict = {p[0]: p[1] for p in all_probs}
                    
                    en_prob = probs_dict.get("en", 0.0)
                    hi_prob = probs_dict.get("hi", 0.0)
                    ur_prob = probs_dict.get("ur", 0.0)
                    hindustani_prob = hi_prob + ur_prob
                    
                    target_prob = en_prob if expected_lang == "en" else hindustani_prob
                    other_prob = hindustani_prob if expected_lang == "en" else en_prob
                    
                    if (en_prob + hindustani_prob) >= 0.05 or target_prob >= 0.10:
                        mismatch_detected = (other_prob - target_prob > mismatch_margin)
                        
                        # Stage 2B - Short Clip Suppressor
                        if mismatch_detected and duration_s < short_clip_threshold_sec and target_prob >= 0.15:
                            mismatch_detected = False
                            
                except Exception as e:
                    print(f"[STT] Acoustic probe error: {e}")

            # Stage 3 - Transcription (No language lock if probe enabled)
            transcribe_lang_param = None if acoustic_probe_enabled else whisper_lang
            segments, info = self.model.transcribe(
                tmp_path,
                beam_size=5,
                language=transcribe_lang_param,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            
            segments_list = list(segments)
            text = " ".join(seg.text.strip() for seg in segments_list).strip()

            if not text or _is_hallucination(text):
                return ""
                
            # Stage 4B - Post-Transcription check
            if acoustic_probe_enabled:
                decoded_lang = getattr(info, "language", None)
                if decoded_lang and decoded_lang != expected_lang and segments_list:
                    _decoded_family = "hi" if decoded_lang in ("hi", "ur") else decoded_lang
                    _expected_family = "hi" if expected_lang in ("hi", "ur") else expected_lang
                    
                    # Ignore short number-like texts
                    import re
                    is_number_like = re.match(r"^[0-9\s,\.]+$", text)
                    if _decoded_family != _expected_family and not is_number_like:
                        print(f"[STT] Mismatch confirmed. Expected {_expected_family}, got {_decoded_family}.")
                        return "[LANGUAGE_MISMATCH]"
                        
            # Stage 8 - Confidence Gate
            avg_logprob = sum(seg.avg_logprob for seg in segments_list) / len(segments_list) if segments_list else -10
            confidence = math.exp(avg_logprob)
            
            min_conf = conf_gates.get("hi" if expected_lang in ("hi", "ur") else "en", 0.45)
            
            if confidence < min_conf:
                print(f"[STT] Rejected: Confidence {confidence:.2f} < {min_conf:.2f} for '{text}'")
                return ""

            print(f"[STT] Transcribed: '{text}' (conf={confidence:.2f})")
            return text

        except Exception as e:
            print(f"[STT] Transcription error: {e}")
            return ""
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def listen(self, prompt: str = "🎤 Listening...",language: str = "en") -> str:
        """
        Full pipeline: record → transcribe → return text.
        Falls back to console input if microphone/Whisper not available.
        """
        if not SOUNDDEVICE_AVAILABLE or not WHISPER_AVAILABLE or self.model is None:
            return input(f"\n{prompt} [Type your response]: ").strip()

        audio = self.record_audio(prompt)
        if audio is None or len(audio) < SAMPLE_RATE * 0.5:
            print("[STT] No audio detected or too short. Please try again.")
            return ""

        # Mic audio is int16 at SAMPLE_RATE (8kHz); normalize to float32
        audio_f32 = audio.astype(np.float32) / 32768.0
        return self.transcribe(audio_f32, language=language)


# Singleton
_stt = None
_stt_model_size = None

def configure_stt(model_size: str) -> None:
    """Call this at startup with the use case config's stt_model_size."""
    global _stt_model_size
    _stt_model_size = model_size

def get_stt() -> WhisperSTT:
    global _stt, _stt_model_size
    if _stt is None:
        model_size = _stt_model_size or os.getenv("STT_MODEL_SIZE", "Systran/faster-whisper-medium")
        _stt = WhisperSTT(model_size=model_size)
    return _stt

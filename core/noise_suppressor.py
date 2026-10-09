"""
noise_suppressor.py — DeepFilterNet real-time noise suppression wrapper.
========================================================================
Inserted into the inbound audio path in server.py immediately after
mulaw_8k_to_float32_16k() and before StreamingSpeechSegmenter.push().

Design:
- Lazy-init: model loads on first call, not at import time.
- DeepFilterNet native SR is 48kHz, so we resample:
      16kHz -> 48kHz -> enhance -> 16kHz
- Controlled by NOISE_SUPPRESSION env var (default: true).
- Graceful fallback: if deepfilternet is missing or fails, audio passes
  through unchanged — no crash, no silent failure.
"""

from __future__ import annotations

import os
import numpy as np

from core.audio_utils import resample_float32

# -- Configuration -------------------------------------------------------------

ENABLED: bool = os.getenv("NOISE_SUPPRESSION", "true").strip().lower() not in ("false", "0", "no")

# -- Lazy singleton state -------------------------------------------------------

_model = None
_df_state = None
_initialized: bool = False
_available: bool = False  # True only when model loaded successfully

_DF_SAMPLE_RATE = 48_000   # DeepFilterNet native SR

# -- Internal init -------------------------------------------------------------

def _lazy_init() -> None:
    global _model, _df_state, _initialized, _available

    if _initialized:
        return
    _initialized = True

    if not ENABLED:
        print("[NS] Noise suppression disabled via NOISE_SUPPRESSION=false")
        return

    try:
        from df.enhance import init_df  # type: ignore[import]
        _model, _df_state, _ = init_df()
        _available = True
        print(f"[NS] DeepFilterNet loaded OK -- native SR={_DF_SAMPLE_RATE}Hz, enabled=True")
    except ImportError:
        print("[NS] deepfilternet not installed (pip install deepfilternet) -- passthrough mode")
    except Exception as exc:
        print(f"[NS] DeepFilterNet init failed ({exc}) -- passthrough mode")


# -- Public API ----------------------------------------------------------------

def denoise(audio_16k: np.ndarray) -> np.ndarray:
    """
    Denoise a 16kHz float32 mono audio array using DeepFilterNet.

    Args:
        audio_16k: float32 numpy array at 16 kHz, shape (N,)

    Returns:
        Denoised float32 numpy array at 16 kHz, same shape as input.
        Falls back to returning the original array on any error.
    """
    _lazy_init()

    if not _available or _model is None or _df_state is None:
        return audio_16k

    if audio_16k is None or audio_16k.size == 0:
        return audio_16k

    try:
        import torch
        from df.enhance import enhance  # type: ignore[import]

        # Upsample 16kHz -> 48kHz (DeepFilterNet native SR)
        audio_48k = resample_float32(audio_16k, 16000, _DF_SAMPLE_RATE)

        # DeepFilterNet expects shape [batch, time] or [1, time]
        tensor_in = torch.from_numpy(audio_48k).unsqueeze(0)   # [1, T]
        enhanced_tensor = enhance(_model, _df_state, tensor_in)
        enhanced_48k = enhanced_tensor.squeeze(0).numpy()

        # Downsample 48kHz -> 16kHz to match the rest of the pipeline
        return resample_float32(enhanced_48k, _DF_SAMPLE_RATE, 16000)

    except Exception as exc:
        # Never crash the call -- silently fall through to original audio
        print(f"[NS] Denoise error ({exc}) -- returning original audio")
        return audio_16k


def is_available() -> bool:
    """Return True if DeepFilterNet is loaded and ready."""
    _lazy_init()
    return _available

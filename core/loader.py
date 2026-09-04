"""
loader.py — Use Case Loader
============================
Reads USE_CASE env var and mounts the right configuration folder.
Every core module receives a UseCaseConfig object — zero hardcoded paths.

To add a new use case:
  1. Create use_cases/<name>/ folder
  2. Add use_case.yaml, flow.json, system_prompt.txt, knowledge_base.json,
     keywords.json, entity_schema.json
  3. Set USE_CASE=<name> in .env
  4. Done — no Python code changes needed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
    print("[LOADER] PyYAML not available; use_case.yaml will be parsed with json fallback.")


# ── UseCaseConfig ─────────────────────────────────────────────────────────────

@dataclass
class UseCaseConfig:
    """
    Resolved configuration for a single use case.
    All file paths are absolute. Callers never construct paths themselves.
    """
    name: str
    base_dir: Path
    raw: dict = field(repr=False)

    # ── File-based properties ─────────────────────────────────────────────────

    @property
    def flow_path(self) -> Path:
        return self.base_dir / self.raw.get("flow", "flow.json")

    @property
    def kb_path(self) -> Path:
        return self.base_dir / self.raw.get("knowledge_base", "knowledge_base.json")

    @property
    def keywords_path(self) -> Path:
        return self.base_dir / self.raw.get("keywords", "keywords.json")

    @property
    def entity_schema_path(self) -> Path:
        return self.base_dir / self.raw.get("entity_schema", "entity_schema.json")

    @property
    def system_prompt_path(self) -> Path:
        return self.base_dir / self.raw.get("system_prompt", "system_prompt.txt")

    @property
    def script_path(self) -> Path:
        return self.base_dir / self.raw.get("script", "script.json")

    @property
    def assets_dir(self) -> Path:
        return self.base_dir / "assets"

    # ── Lazy-loaded content ────────────────────────────────────────────────────

    def get_system_prompt(self) -> str:
        """Read system_prompt.txt for this use case."""
        path = self.system_prompt_path
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return ""

    def get_flow(self) -> dict:
        """Read and parse flow.json for this use case."""
        with open(self.flow_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_kb_path_str(self) -> str:
        return str(self.kb_path)

    def get_keywords(self) -> dict:
        """Read and parse keywords.json for this use case."""
        with open(self.keywords_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_entity_schema(self) -> dict:
        """Read and parse entity_schema.json for this use case."""
        with open(self.entity_schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_hooks_module(self):
        """
        Optionally load a use-case-specific hooks.py plugin.
        Returns None if no hooks.py exists for this use case.
        Hooks contain complex action logic (e.g. HTTP calls, WhatsApp, NPCI).
        """
        hooks_path = self.base_dir / "hooks.py"
        if not hooks_path.exists():
            return None
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            f"use_cases.{self.name}.hooks", hooks_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # ── Scalar config helpers ─────────────────────────────────────────────────

    @property
    def display_name(self) -> str:
        return self.raw.get("display_name", self.name)

    @property
    def server_port(self) -> int:
        return int(self.raw.get("server_port", 8767))

    @property
    def filler_phrases(self) -> list[str]:
        return self.raw.get("filler_phrases", [
            "Just a moment.",
            "Let me check that.",
            "One second.",
        ])

    @property
    def session_ttl_seconds(self) -> int:
        return int(self.raw.get("session_ttl_seconds", 3600))

    @property
    def cleanup_interval_seconds(self) -> int:
        return int(self.raw.get("cleanup_interval_seconds", 300))

    # ── Asterisk ARI ─────────────────────────────────────────────────────────

    @property
    def asterisk_host(self) -> str:
        return self.raw.get("asterisk", {}).get("host") or os.getenv("ASTERISK_HOST", "127.0.0.1")

    @property
    def asterisk_ari_port(self) -> str:
        return str(self.raw.get("asterisk", {}).get("ari_port") or os.getenv("ASTERISK_ARI_PORT", "8088"))

    @property
    def asterisk_app_name(self) -> str:
        return self.raw.get("asterisk", {}).get("app_name") or os.getenv("APP_NAME", "ai_agent")

    # ── STT (faster-whisper) ──────────────────────────────────────────────────

    @property
    def stt_model_size(self) -> str:
        return self.raw.get("stt", {}).get("model_size", "Systran/faster-whisper-medium")

    @property
    def stt_language(self) -> str:
        return self.raw.get("stt", {}).get("language", "auto")

    @property
    def stt_min_audio_duration_s(self) -> float:
        return float(self.raw.get("stt", {}).get("min_audio_duration_s", 1.0))

    # ── TTS (Indic-Mio) ───────────────────────────────────────────────────────

    @property
    def tts_url(self) -> str:
        return self.raw.get("tts", {}).get("url") or os.getenv("MIO_TTS_URL", "http://localhost:8001/v1/tts")

    @property
    def tts_speaker_wav(self) -> str | None:
        rel = self.raw.get("tts", {}).get("speaker_wav", "")
        if rel:
            candidate = self.base_dir / rel
            if candidate.exists():
                return str(candidate)
        # fall back to env
        return os.getenv("F5_SPEAKER_WAV", "")

    @property
    def tts_max_chunk_chars(self) -> int:
        return int(self.raw.get("tts", {}).get("max_chunk_chars", 250))

    @property
    def tts_preset_id(self) -> str:
        return self.raw.get("tts", {}).get("preset_id", "my_speaker_saavi")

    # ── LLM (Qwen via vLLM) ──────────────────────────────────────────────────

    @property
    def llm_backend(self) -> str:
        return self.raw.get("llm", {}).get("backend") or os.getenv("LLM_BACKEND", "vllm")

    @property
    def llm_base_url(self) -> str:
        return self.raw.get("llm", {}).get("base_url") or os.getenv("VLLM_BASE_URL", "http://localhost:8002/v1")

    @property
    def llm_model(self) -> str:
        return self.raw.get("llm", {}).get("model") or os.getenv("VLLM_MODEL", "qwen2.5:1.5b")

    @property
    def llm_max_tokens(self) -> int:
        return int(self.raw.get("llm", {}).get("max_tokens", 150))

    @property
    def llm_temperature(self) -> float:
        return float(self.raw.get("llm", {}).get("temperature", 0.3))

    # ── RAG ───────────────────────────────────────────────────────────────────

    @property
    def rag_threshold(self) -> float:
        return float(self.raw.get("rag", {}).get("threshold", 0.55))

    @property
    def rag_max_results(self) -> int:
        return int(self.raw.get("rag", {}).get("max_results", 3))

    # ── Extra KB dirs (for supplementary JSON KB folders) ────────────────────

    @property
    def extra_kb_dirs(self) -> list[str]:
        """Optional list of extra KB directory paths relative to use_case base."""
        dirs = self.raw.get("extra_kb_dirs", [])
        return [str(self.base_dir / d) for d in dirs]


# ── Loader ────────────────────────────────────────────────────────────────────

def _parse_yaml_or_json(path: Path) -> dict:
    """Parse use_case.yaml using PyYAML, fallback to JSON if yaml unavailable."""
    text = path.read_text(encoding="utf-8")
    if _YAML_AVAILABLE:
        return yaml.safe_load(text) or {}
    # crude fallback: try JSON (won't work for proper YAML, but covers simple cases)
    try:
        return json.loads(text)
    except Exception:
        raise RuntimeError(
            f"[LOADER] PyYAML is not installed and {path} is not valid JSON. "
            "Install PyYAML: pip install pyyaml"
        )


# ── Singleton cache ───────────────────────────────────────────────────────────
_cache: dict[str, UseCaseConfig] = {}

_PLATFORM_ROOT = Path(__file__).resolve().parent.parent  # voice-platform/


def load_use_case(name: str | None = None) -> UseCaseConfig:
    """
    Load a use case by name (or from USE_CASE env var).

    Args:
        name: use case folder name under use_cases/.
              Defaults to USE_CASE env var, then 'admissions'.

    Returns:
        UseCaseConfig with all paths resolved.

    Raises:
        FileNotFoundError: if the use case folder or use_case.yaml is missing.
    """
    name = name or os.getenv("USE_CASE", "admissions")

    if name in _cache:
        return _cache[name]

    use_cases_root = _PLATFORM_ROOT / "use_cases"
    base = use_cases_root / name

    if not base.exists():
        available = [d.name for d in use_cases_root.iterdir() if d.is_dir()] if use_cases_root.exists() else []
        raise FileNotFoundError(
            f"[LOADER] Use case '{name}' not found at {base}. "
            f"Available: {available}"
        )

    config_file = base / "use_case.yaml"
    if not config_file.exists():
        raise FileNotFoundError(
            f"[LOADER] Missing use_case.yaml for '{name}' at {config_file}"
        )

    raw = _parse_yaml_or_json(config_file)
    cfg = UseCaseConfig(name=name, base_dir=base, raw=raw)

    # Validate required files exist
    if name in ("admissions", "zipnach"):
        required_files = [
            ("script.json", cfg.script_path),
        ]
    else:
        required_files = [
            ("flow.json", cfg.flow_path),
            ("system_prompt.txt", cfg.system_prompt_path),
            ("knowledge_base.json", cfg.kb_path),
            ("keywords.json", cfg.keywords_path),
            ("entity_schema.json", cfg.entity_schema_path),
        ]

    for attr, path in required_files:
        if not path.exists():
            print(f"[LOADER] WARNING: Required file missing for use case '{name}': {path}")

    _cache[name] = cfg
    print(f"[LOADER] Loaded use case: '{name}' from {base}")
    return cfg


def reload_use_case(name: str | None = None) -> UseCaseConfig:
    """Force-reload a use case, clearing the cache."""
    name = name or os.getenv("USE_CASE", "admissions")
    _cache.pop(name, None)
    return load_use_case(name)

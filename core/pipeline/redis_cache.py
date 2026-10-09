"""
redis_cache.py — Multi-Use Case & Multi-Voice Redis Caching Layer (Strategy B: DB Indexing per .env)
===================================================================================================
Provides asynchronous and synchronous Redis caching for:
1. Multi-voice TTS audio caching (partitioned by REDIS_DB index per use-case .env + voice_id)
2. Flow / Session memory storage
3. FAQ / RAG response caching
"""

import os
import hashlib
import unicodedata
import logging
from typing import Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger("redis_cache")


class RedisCacheManager:
    def __init__(self):
        self.reload_config()

    def reload_config(self):
        """Reload configuration from environment variables (useful after loading use case .env)."""
        self.use_case = os.getenv("USE_CASE", "default").strip().lower()
        self.host = os.getenv("REDIS_HOST", "localhost").strip()
        self.port = int(os.getenv("REDIS_PORT", "6379"))
        self.db = int(os.getenv("REDIS_DB", "0"))
        self.prefix = os.getenv("REDIS_PREFIX", self.use_case).strip().lower()
        self.password = os.getenv("REDIS_PASSWORD", None)

        self._sync_client = None
        if REDIS_AVAILABLE:
            try:
                self._sync_client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password if self.password else None,
                    socket_connect_timeout=2,
                )
                self._sync_client.ping()
                print(f"[REDIS] OK Connected to Redis {self.host}:{self.port} | DB {self.db} | Prefix '{self.prefix}'")
            except Exception as e:
                print(f"[REDIS] Notice: Redis offline on DB {self.db} ({e}). Using local fallback.")
                self._sync_client = None

    def _tts_cache_key(self, text: str, language: str, voice_id: str) -> str:
        clean = unicodedata.normalize("NFKD", f"{language.strip().lower()}|{voice_id.strip().lower()}|{text.strip()}")
        text_hash = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        return f"{self.prefix}:tts:{voice_id.strip().lower()}:{text_hash}"

    def get_tts_audio(self, text: str, language: str = "en", voice_id: str = "default") -> Optional[bytes]:
        if not self._sync_client:
            return None
        key = self._tts_cache_key(text, language, voice_id)
        try:
            val = self._sync_client.get(key)
            if val:
                print(f"[REDIS-CACHE DB{self.db}] HIT for voice='{voice_id}' ({len(val)} bytes)")
                return val
        except Exception as e:
            logger.warning(f"[REDIS] Get failed for {key}: {e}")
        return None

    def set_tts_audio(self, text: str, language: str, voice_id: str, audio_bytes: bytes, ttl_seconds: int = 604800) -> bool:
        if not self._sync_client or not audio_bytes:
            return False
        key = self._tts_cache_key(text, language, voice_id)
        try:
            self._sync_client.set(key, audio_bytes, ex=ttl_seconds)
            print(f"[REDIS-CACHE DB{self.db}] STORED voice='{voice_id}' ({len(audio_bytes)} bytes, TTL={ttl_seconds}s)")
            return True
        except Exception as e:
            logger.warning(f"[REDIS] Set failed for {key}: {e}")
            return False


redis_cache = RedisCacheManager()

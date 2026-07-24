"""
memory.py — Generic conversation memory for Voice AI.
======================================================
Entity extractor schema is loaded from the use-case's entity_schema.json.
Lead profile fields are dynamic — no hardcoded domain-specific logic.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Optional

# ── Module-level schema (set via configure_memory) ────────────────────────────
_ENTITY_SCHEMA: dict = {}   # parsed entity_schema.json

def configure_memory(entity_schema: dict) -> None:
    """Call at startup with the use-case entity schema."""
    global _ENTITY_SCHEMA
    _ENTITY_SCHEMA = entity_schema
    print(f"[Memory] Entity schema loaded: {list(entity_schema.get('fields', {}).keys())}")


# ── Generic Entity Extractor ──────────────────────────────────────────────────

class GenericEntityExtractor:
    """
    Extracts entities from user utterances using rules from entity_schema.json.

    Schema format (entity_schema.json):
    {
      "fields": {
        "field_name": {
          "type": "boolean|enum|phone|datetime|text",
          "positive_phrases": [...],   # for boolean
          "negative_phrases": [...],   # for boolean
          "values": [...],             # for enum — keyword match
          "regex": "...",              # for phone/pattern
          "keywords": {...}            # for enum — {value: [keywords]}
        }
      }
    }
    """

    PHONE_RE = re.compile(r"\b(\+?91[\s-]?)?[6-9]\d{9}\b")

    @classmethod
    def extract(cls, text: str) -> dict:
        """
        Returns a dict of detected fields from a single utterance.
        Only definitive signals are returned.
        """
        entities: dict = {}
        t = text.lower()
        schema = _ENTITY_SCHEMA.get("fields", {})

        for field_name, field_def in schema.items():
            field_type = field_def.get("type", "text")

            if field_type == "phone":
                regex = field_def.get("regex")
                if regex:
                    m = re.search(regex, text)
                else:
                    m = cls.PHONE_RE.search(text)
                if m:
                    digits = re.sub(r"\D", "", m.group(0))
                    if len(digits) >= 10:
                        entities[field_name] = digits[-10:]

            elif field_type == "boolean":
                pos = field_def.get("positive_phrases", [])
                neg = field_def.get("negative_phrases", [])
                if any(k in t for k in pos):
                    entities[field_name] = True
                elif any(k in t for k in neg):
                    entities[field_name] = False

            elif field_type == "enum":
                # Try keyword matching per value
                kw_map = field_def.get("keywords", {})
                if kw_map:
                    for val, keywords in kw_map.items():
                        if any(k in t for k in keywords):
                            entities[field_name] = val
                            break
                else:
                    # Fallback: check if any value appears in text
                    for val in field_def.get("values", []):
                        if val.lower() in t:
                            entities[field_name] = val
                            break

            elif field_type == "text":
                # Simple keyword-triggered text capture (use-case defines what to look for)
                kw_map = field_def.get("keywords", {})
                for val, keywords in kw_map.items():
                    if any(k in t for k in keywords):
                        entities[field_name] = val
                        break

        return entities


# ── Lead Profile (generic, field-order driven by entity_schema) ───────────────

class LeadProfile:
    """
    Generic lead/session profile.
    Fields are dynamically populated from the entity_schema.
    """

    def __init__(self) -> None:
        self._fields: dict = {}
        # Build initial field set from schema
        schema = _ENTITY_SCHEMA.get("fields", {})
        for field_name in schema:
            self._fields[field_name] = None

    def get(self, field: str, default=None):
        return self._fields.get(field, default)

    def set(self, field: str, value) -> None:
        self._fields[field] = value

    def update_from_entities(self, entities: dict) -> list[str]:
        """Merge extracted entities. Returns list of updated field names."""
        updated = []
        for field, value in entities.items():
            if field not in self._fields:
                self._fields[field] = value
                updated.append(field)
            elif self._fields[field] is None:
                self._fields[field] = value
                updated.append(field)
            elif self._fields[field] != value:
                # Only update if schema allows overwrite
                schema_field = _ENTITY_SCHEMA.get("fields", {}).get(field, {})
                if schema_field.get("allow_overwrite", True):
                    self._fields[field] = value
                    updated.append(field)
        return updated

    def as_dict(self) -> dict:
        return dict(self._fields)

    def next_missing_field(self) -> Optional[str]:
        """Return name of next required-but-missing field, per schema order."""
        schema = _ENTITY_SCHEMA.get("fields", {})
        required_order = _ENTITY_SCHEMA.get("required_order", list(schema.keys()))
        for field in required_order:
            if self._fields.get(field) is None:
                return field
        return None


# ── Conversation Memory ───────────────────────────────────────────────────────

class ConversationMemory:
    """
    Tracks conversation history + lead profile for one session.
    Domain-agnostic — receives entity schema from use-case config.
    """

    def __init__(self, session_id: str, ttl_seconds: int = 3600):
        self.session_id   = session_id
        self.ttl_seconds  = ttl_seconds
        self.created_at   = time.time()
        self.last_active  = time.time()

        self.history:          list[dict] = []
        self.lead:             LeadProfile = LeadProfile()
        self.detected_language: str = "en"
        self.current_state:    str = ""
        self.turn_count:       int = 0
        self.topics_discussed: list[str] = []

    def add_turn(self, user_text: str, bot_response: str) -> None:
        self.last_active = time.time()
        self.turn_count += 1
        if user_text:
            self.history.append({"role": "user", "content": user_text})
        if bot_response:
            self.history.append({"role": "assistant", "content": bot_response})

    def update_entities(self, text: str) -> list[str]:
        """Extract and merge entities from text. Returns updated field names."""
        entities = GenericEntityExtractor.extract(text)
        return self.lead.update_from_entities(entities)

    def get_recent_history(self, n: int = 6) -> list[dict]:
        return self.history[-n:] if len(self.history) > n else list(self.history)

    def get_session_info(self) -> dict:
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "language":   self.detected_language,
            "state":      self.current_state,
            **self.lead.as_dict(),
        }

    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > self.ttl_seconds


# ── Memory Manager (multi-session) ────────────────────────────────────────────

class MemoryManager:
    """Manages conversation memory across all active sessions."""

    def __init__(self):
        self._sessions: dict[str, ConversationMemory] = {}

    def get_or_create(
        self,
        session_id: str,
        ttl_seconds: int = 3600,
    ) -> ConversationMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(
                session_id=session_id,
                ttl_seconds=ttl_seconds,
            )
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[ConversationMemory]:
        return self._sessions.get(session_id)

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def cleanup_expired(self, ttl_seconds: int = 3600) -> int:
        expired = [
            sid for sid, mem in self._sessions.items()
            if (time.time() - mem.last_active) > ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            print(f"[Memory] Cleaned up {len(expired)} expired sessions")
        return len(expired)

    def active_sessions(self) -> int:
        return len(self._sessions)


# ── Singleton ─────────────────────────────────────────────────────────────────
memory_manager = MemoryManager()

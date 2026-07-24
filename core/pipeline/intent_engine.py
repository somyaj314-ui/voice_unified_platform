"""
intent_engine.py — Generic intent classifier.
==============================================
Keywords are loaded from the use-case's keywords.json via configure_intent().
Universal intents (repeat, wait, exit, affirm, deny, phone) work across all use cases.
Use-case-specific intents are declared in entity_schema.json.
"""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from typing import Optional


# ── Intent name constants ─────────────────────────────────────────────────────
class IntentName:
    # Global overrides
    WRONG_NUMBER   = "wrong_number"
    WRONG_PERSON   = "wrong_person"
    DND            = "dnd"
    BUSY_CALLBACK  = "busy_callback"
    HUMAN_HANDOFF  = "human_handoff"
    EXIT           = "exit"

    # Conversational signals
    REPEAT         = "repeat"
    WAIT           = "wait"
    HELP           = "help"

    # Generic positive/negative
    AFFIRM         = "affirm"
    DENY           = "deny"

    # Generic slot signals (interpreted by flow per state)
    PHONE_NUMBER   = "phone_number"
    TOPIC_QUERY    = "topic_query"
    UNKNOWN        = "unknown"


@dataclass
class Intent:
    name:       str
    confidence: float = 1.0
    data:       dict  = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}


_PHONE_RE = re.compile(r"\b(\+?91[\s\-]?)?[6-9]\d{9}\b")

# ── Module-level keyword tables ───────────────────────────────────────────────
_KW: dict = {}
_WRONG_NUMBER_KW:   list = []
_WRONG_PERSON_KW:   list = []
_DND_KW:            list = []
_CALLBACK_KW:       list = []
_HUMAN_KW:          list = []
_EXIT_KW:           list = []
_REPEAT_KW:         list = []
_WAIT_KW:           list = []
_HELP_KW:           list = []
_AFFIRM_KW:         set  = set()
_DENY_KW:           set  = set()
_TOPIC_KW:          dict = {}
_CUSTOM_KW:         dict = {}   # use-case-specific keyword lists
_LLM_CATEGORY_MAP:  dict = {}
_AMBIGUOUS_SHORT:   set  = set()


def configure_intent(keywords: dict) -> None:
    """Load all keyword tables from the use-case's keywords.json."""
    global _KW, _WRONG_NUMBER_KW, _WRONG_PERSON_KW, _DND_KW
    global _CALLBACK_KW, _HUMAN_KW, _EXIT_KW, _REPEAT_KW, _WAIT_KW, _HELP_KW
    global _AFFIRM_KW, _DENY_KW, _TOPIC_KW, _CUSTOM_KW, _LLM_CATEGORY_MAP, _AMBIGUOUS_SHORT

    _KW              = keywords
    _WRONG_NUMBER_KW = _KW.get("WRONG_NUMBER_KEYWORDS", [])
    _WRONG_PERSON_KW = _KW.get("WRONG_PERSON_KEYWORDS", [])
    _DND_KW          = _KW.get("DND_KEYWORDS", [])
    _CALLBACK_KW     = _KW.get("CALL_LATER_KEYWORDS", [])
    _HUMAN_KW        = _KW.get("HUMAN_AGENT_KEYWORDS", [])
    _EXIT_KW         = _KW.get("EXIT_KEYWORDS", [])
    _REPEAT_KW       = _KW.get("REPEAT_KEYWORDS", [])
    _WAIT_KW         = _KW.get("WAIT_KEYWORDS", [])
    _HELP_KW         = _KW.get("HELP_KEYWORDS", [])
    _AFFIRM_KW       = set(_KW.get("WEAK_POSITIVES", [])) | set(_KW.get("STRONG_COMPLETION", []))
    _DENY_KW         = set(_KW.get("NEGATIVE_KEYWORDS", []))
    _TOPIC_KW        = _KW.get("TOPIC_ROUTE_KEYWORDS", {})
    _CUSTOM_KW       = _KW.get("CUSTOM_INTENTS", {})   # use-case-specific intents
    _LLM_CATEGORY_MAP = _KW.get("LLM_CATEGORY_MAP", {})
    _AMBIGUOUS_SHORT  = set(_KW.get("AMBIGUOUS_SHORT", []))
    print(f"[IntentEngine] Loaded keywords: {len(_KW)} categories")


# ── Public API ────────────────────────────────────────────────────────────────

def classify(text: str, student_name: str = "", current_state: str = "") -> list[Intent]:
    """
    Classify a user utterance into one or more Intents.
    Returns a list ordered by priority (most important first).
    """
    if not text or not text.strip():
        return [Intent(IntentName.UNKNOWN)]

    t      = text.lower().strip()
    words  = set(t.split())
    intents: list[Intent] = []

    # ── Priority 1: Global overrides ─────────────────────────────────────────
    if _WRONG_NUMBER_KW and any(k in t for k in _WRONG_NUMBER_KW):
        intents.append(Intent(IntentName.WRONG_NUMBER))

    if not intents and _WRONG_PERSON_KW:
        if any(k in t for k in _WRONG_PERSON_KW):
            intents.append(Intent(IntentName.WRONG_PERSON))
        elif student_name and ("not " + student_name.lower()) in t:
            intents.append(Intent(IntentName.WRONG_PERSON))

    if _DND_KW and any(k in t for k in _DND_KW):
        intents.append(Intent(IntentName.DND))

    if _HUMAN_KW and any(k in t for k in _HUMAN_KW):
        intents.append(Intent(IntentName.HUMAN_HANDOFF))

    if _CALLBACK_KW and any(k in t for k in _CALLBACK_KW):
        intents.append(Intent(IntentName.BUSY_CALLBACK))
    elif _EXIT_KW and any(k in t for k in _EXIT_KW):
        intents.append(Intent(IntentName.EXIT))

    # ── Priority 2: Conversational signals ───────────────────────────────────
    if _REPEAT_KW and any(k in t for k in _REPEAT_KW):
        intents.append(Intent(IntentName.REPEAT))
    if _WAIT_KW and any(k in t for k in _WAIT_KW):
        intents.append(Intent(IntentName.WAIT))
    if _HELP_KW and any(k in t for k in _HELP_KW):
        intents.append(Intent(IntentName.HELP))

    # ── Priority 3: Use-case-specific custom intents ──────────────────────────
    for intent_name, kw_list in _CUSTOM_KW.items():
        if isinstance(kw_list, list) and any(k in t for k in kw_list):
            intents.append(Intent(intent_name))

    # ── Priority 4: Phone number ──────────────────────────────────────────────
    phone = _extract_phone(text)
    if phone:
        intents.append(Intent(IntentName.PHONE_NUMBER, data={"phone": phone}))

    # ── Priority 5: Affirm / Deny ─────────────────────────────────────────────
    _check_affirm_deny(t, intents, words)

    # ── Priority 6: Topic query ───────────────────────────────────────────────
    topic = _detect_topic(t)
    if topic:
        intents.append(Intent(IntentName.TOPIC_QUERY, data={"topic": topic}))

    # ── LLM fallback ─────────────────────────────────────────────────────────
    LOW_CONF = 0.6
    needs_llm = (
        not intents
        or all(i.name == IntentName.UNKNOWN for i in intents)
        or any(i.confidence < LOW_CONF for i in intents)
    )
    if needs_llm:
        llm_intent = _llm_fallback(text, current_state)
        if llm_intent:
            intents = [i for i in intents if i.confidence >= LOW_CONF]
            intents.insert(0, llm_intent)

    return intents or [Intent(IntentName.UNKNOWN)]


def primary_intent(text: str, student_name: str = "", current_state: str = "") -> Intent:
    return classify(text, student_name, current_state)[0]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_phone(text: str) -> Optional[str]:
    m = _PHONE_RE.search(text)
    if not m:
        return None
    digits = re.sub(r"\D", "", m.group(0))
    return digits[-10:] if len(digits) >= 10 else None


def _detect_topic(t: str) -> Optional[str]:
    for topic, kws in _TOPIC_KW.items():
        if any(k in t for k in kws):
            return topic
    return None


def _check_affirm_deny(t: str, intents: list[Intent], words: set) -> None:
    """Only add AFFIRM/DENY when no stronger specific intent is already present."""
    has_specific = any(
        i.name not in {IntentName.UNKNOWN, IntentName.TOPIC_QUERY, IntentName.PHONE_NUMBER}
        and i.name not in {IntentName.REPEAT, IntentName.WAIT, IntentName.HELP}
        for i in intents
    )
    is_short = len(words) <= 3
    if not has_specific:
        if t in _AFFIRM_KW or (is_short and any(w in _AFFIRM_KW for w in words)):
            intents.append(Intent(IntentName.AFFIRM, confidence=0.85))
        if t in _DENY_KW or (is_short and any(w in _DENY_KW for w in words)):
            intents.append(Intent(IntentName.DENY, confidence=0.85))


def _llm_fallback(text: str, current_state: str = "") -> Optional[Intent]:
    try:
        from core.pipeline.llm_engine import get_llm
        llm      = get_llm()
        category = llm.classify_intent(text, current_state)
        intent_name = _LLM_CATEGORY_MAP.get(category, IntentName.UNKNOWN)
        if intent_name != IntentName.UNKNOWN:
            print(f"[IntentLLM] '{text[:50]}' → {category} → {intent_name}")
            return Intent(intent_name, confidence=0.75)
        return None
    except Exception as e:
        print(f"[IntentLLM] fallback error: {e}")
        return None

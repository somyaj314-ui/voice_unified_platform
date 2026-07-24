"""
llm_engine.py — Generic LLM engine (Qwen 2.5 1.5B via vLLM).
=============================================================
System prompt is loaded from the use case config — NOT hardcoded here.
Every use case provides its own system_prompt.txt.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from typing import Optional

try:
    import ollama
except Exception:
    ollama = None

from dotenv import load_dotenv

load_dotenv()

# ── Module-level config (set once at startup via configure_llm) ───────────────
_SYSTEM_PROMPT: str = ""        # loaded from use_case/system_prompt.txt
_LLM_BACKEND: str = os.getenv("LLM_BACKEND", "vllm").strip().lower()
_VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8002/v1").rstrip("/")
_VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "EMPTY")
_VLLM_MODEL: str = os.getenv("VLLM_MODEL", "qwen2.5:1.5b")
_INTENT_MODEL: str = os.getenv("VLLM_INTENT_MODEL", _VLLM_MODEL)
_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "150"))
_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# ── Category map for LLM intent classifier (generic) ─────────────────────────
_DEFAULT_CATEGORY_MAP = {
    "CONFIRMATION": "affirm",
    "REJECTION":    "deny",
    "QUERY":        "unknown",
    "REPEAT":       "repeat",
    "WAIT":         "wait",
    "GOODBYE":      "exit",
    "EXIT":         "exit",
    "NOISE":        "unknown",
}
_LLM_CATEGORY_MAP: dict = {}


def configure_llm(
    system_prompt: str,
    *,
    backend: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    category_map: dict | None = None,
) -> None:
    """
    Call at startup with the use-case config.
    Resets the LLM singleton so it picks up new settings.
    """
    global _SYSTEM_PROMPT, _LLM_BACKEND, _VLLM_BASE_URL, _VLLM_MODEL
    global _MAX_TOKENS, _TEMPERATURE, _LLM_CATEGORY_MAP, _llm

    _SYSTEM_PROMPT = system_prompt
    if backend:
        _LLM_BACKEND = backend
    if base_url:
        _VLLM_BASE_URL = base_url.rstrip("/")
    if model:
        _VLLM_MODEL = model
    if max_tokens is not None:
        _MAX_TOKENS = max_tokens
    if temperature is not None:
        _TEMPERATURE = temperature
    _LLM_CATEGORY_MAP = category_map or _DEFAULT_CATEGORY_MAP
    _llm = None   # force re-init


# ── Engine ────────────────────────────────────────────────────────────────────

class LLMEngine:
    """
    Generic LLM wrapper — vLLM (primary) or Ollama (fallback).
    The system prompt is injected per-call from the use-case config,
    NOT embedded in this class.
    """

    def __init__(self):
        self.backend       = _LLM_BACKEND if _LLM_BACKEND in {"vllm", "ollama"} else "vllm"
        self.vllm_base_url = _VLLM_BASE_URL
        self.vllm_api_key  = _VLLM_API_KEY
        self.vllm_model    = _VLLM_MODEL
        self.intent_model  = _INTENT_MODEL
        self.ollama_model  = _VLLM_MODEL
        print(
            f"[LLM] backend={self.backend} "
            f"model={self.vllm_model if self.backend == 'vllm' else self.ollama_model}"
        )

    # ── Internal dispatch ─────────────────────────────────────────────────────

    def _call_chat_backend(
        self,
        messages: list,
        model_name: str,
        temperature: float,
        max_tokens: int,
        response_format: str | None = None,
    ) -> str:
        if self.backend == "vllm":
            return self._call_vllm(messages, model_name, temperature, max_tokens, response_format)
        if ollama is None:
            raise RuntimeError("Ollama library is not installed")
        kwargs: dict = {
            "model": model_name,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if response_format == "json":
            kwargs["format"] = "json"
        response = ollama.chat(**kwargs)
        return str(response["message"]["content"]).strip()

    def _call_vllm(
        self,
        messages: list,
        model_name: str,
        temperature: float,
        max_tokens: int,
        response_format: str | None = None,
    ) -> str:
        payload: dict = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json"}
        if self.vllm_api_key and self.vllm_api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.vllm_api_key}"

        request = urllib.request.Request(
            f"{self.vllm_base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:
                data = json.load(resp)
        except Exception as exc:
            raise RuntimeError(f"vLLM request failed: {exc}") from exc

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("vLLM returned no choices")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return str(content).strip()

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_response(
        self,
        user_message: str,
        conversation_history: list,
        session_info: dict | None = None,
        context: str = "",
    ) -> str:
        """
        Generate a voice-ready response using the use-case system prompt.
        system_prompt is read from the module-level _SYSTEM_PROMPT
        which is set by configure_llm() at startup.
        """
        system_prompt = _SYSTEM_PROMPT or "You are a helpful voice assistant."

        if session_info:
            system_prompt += "\n\n[SESSION CONTEXT]"
            name = session_info.get("student_name") or session_info.get("customer_name")
            if name:
                system_prompt += f"\n- Name: {name}"
            for key, label in [
                ("graduation_status", "Graduation"),
                ("exam_name", "Exam given"),
                ("specialization", "Interested specialization"),
                ("counselling_mode", "Counselling preference"),
                ("state", "Current flow step"),
            ]:
                if session_info.get(key):
                    system_prompt += f"\n- {label}: {session_info[key]}"
            system_prompt += "\n\nIMPORTANT: Keep the answer to 1-2 sentences. No bullet points. Voice-ready."

        messages = [{"role": "system", "content": system_prompt}]
        recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        messages.extend(recent_history)

        if context:
            augmented = f"[Knowledge base context: {context}]\n\nUser said: {user_message}"
        else:
            augmented = user_message
        messages.append({"role": "user", "content": augmented})

        try:
            reply = self._call_chat_backend(
                messages=messages,
                model_name=self.vllm_model if self.backend == "vllm" else self.ollama_model,
                temperature=_TEMPERATURE,
                max_tokens=_MAX_TOKENS,
            )
            reply = reply.replace("*", "").replace("#", "").replace("`", "")
            print(f"[LLM] Generated: '{reply[:80]}'")
            return reply
        except Exception as e:
            print(f"[LLM] Error: {e}")
            return "Let me get you the right information. Could you please hold on?"

    def classify_intent(self, user_text: str, current_state: str, history: list | None = None) -> str:
        """
        Classify user intent when keywords fail.
        Returns one of: CONFIRMATION, REJECTION, QUERY, REPEAT, WAIT, GOODBYE, EXIT, NOISE
        """
        history_context = ""
        if history:
            recent = history[-4:]
            history_context = "Recent conversation:\n"
            for msg in recent:
                role = "Agent" if msg["role"] == "assistant" else "User"
                history_context += f"{role}: {msg['content']}\n"

        prompt = f"""Classify the user's intent for a voice assistant.
Current State: {current_state}
{history_context}
User said: "{user_text}"

Categories:
- CONFIRMATION: yes, done, completed, haan, hogya, kar liya
- REJECTION: no, cancel, nahi, nhi, abhi nahi
- QUERY: user asks a question or needs info
- REPEAT: asks to repeat (phir se, dobara, repeat)
- WAIT: asks to wait (ruko, hold on, one minute)
- GOODBYE: wants to end call (bye, dhanyavaad)
- EXIT: wants to stop and do later (baad me, phir kabhi)
- NOISE: irrelevant noise or greetings

Output ONLY the category name."""
        try:
            result = self._call_chat_backend(
                messages=[
                    {"role": "system", "content": "You are a precise intent classifier. Output only one word."},
                    {"role": "user", "content": prompt},
                ],
                model_name=self.intent_model if self.backend == "vllm" else self.ollama_model,
                temperature=0,
                max_tokens=10,
            ).strip().upper()
            for cat in ["CONFIRMATION", "REJECTION", "QUERY", "REPEAT", "WAIT", "GOODBYE", "EXIT"]:
                if cat in result:
                    print(f"[LLM-Classifier] '{user_text[:50]}' → {cat}")
                    return cat
            return "NOISE"
        except Exception as e:
            print(f"[LLM-Classifier] Error: {e}")
            return "NOISE"

    def classify_slot(self, current_state: str, user_text: str, last_bot_message: str = "") -> dict:
        """
        Lightweight JSON slot classifier for FSM fallback.
        Schema is defined per-state in flow.json via 'llm_slot_schema'.
        """
        state = (current_state or "").strip().upper()
        utterance = (user_text or "").strip()
        if not utterance:
            return {"intent": "unclear"}

        prompt = f"""State: {state}
Agent Question: {last_bot_message}
User Response: {utterance}

Analyze the response and return a JSON object with an 'intent' field.
Use 'unclear' if you cannot confidently classify.
Output ONLY valid JSON."""
        try:
            raw = self._call_chat_backend(
                messages=[
                    {"role": "system", "content": "You are a strict JSON intent classifier. Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model_name=self.intent_model if self.backend == "vllm" else self.ollama_model,
                temperature=0,
                max_tokens=60,
                response_format="json",
            ).strip()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
                data = json.loads(match.group(0)) if match else {"intent": "unclear"}
            if not isinstance(data, dict):
                data = {"intent": "unclear"}
            data["intent"] = str(data.get("intent") or "unclear").strip().lower()
            print(f"[LLM-Slot] state={state} → {data}")
            return data
        except Exception as e:
            print(f"[LLM-Slot] Error: {e}")
            return {"intent": "unclear"}

    def is_available(self) -> bool:
        return True


# ── Singleton ─────────────────────────────────────────────────────────────────
_llm: Optional[LLMEngine] = None


def get_llm() -> LLMEngine:
    global _llm
    if _llm is None:
        _llm = LLMEngine()
    return _llm

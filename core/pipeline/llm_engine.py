"""
llm_engine.py — Generic LLM engine (strictly for Intent Classification and Routing).
======================================================================================
System prompt is loaded from the use case config and injected into intent classifiers.
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

# ── Module-level config ───────────────
_SYSTEM_PROMPT: str = ""        # loaded from use_case/system_prompt.txt
_LLM_BACKEND: str = os.getenv("LLM_BACKEND", "vllm").strip().lower()
_VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8002/v1").rstrip("/")
_VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "EMPTY")
_VLLM_MODEL: str = os.getenv("VLLM_MODEL", "qwen2.5:1.5b")
_INTENT_MODEL: str = os.getenv("VLLM_INTENT_MODEL", _VLLM_MODEL)


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
    Resets the LLM singleton so it picks up new settings and system prompt.
    """
    global _SYSTEM_PROMPT, _LLM_BACKEND, _VLLM_BASE_URL, _VLLM_MODEL, _llm

    _SYSTEM_PROMPT = system_prompt
    if backend:
        _LLM_BACKEND = backend
    if base_url:
        _VLLM_BASE_URL = base_url.rstrip("/")
    if model:
        _VLLM_MODEL = model
    _llm = None   # force re-init


# ── Engine ────────────────────────────────────────────────────────────────────

class LLMEngine:
    """
    Generic LLM wrapper for classification/routing.
    The system prompt is injected per-call from the use-case config.
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

    # ── Intent Classification APIs ────────────────────────────────────────────

    def classify_intent(self, user_text: str, current_state: str, history: list = None) -> str:
        """
        Use local LLM to classify user intent when keywords fail.
        Returns one of: CONFIRMATION, REJECTION, QUERY, REPEAT, WAIT, GOODBYE, EXIT, NOISE
        """
        history_context = ""
        if history:
            recent = history[-4:]
            history_context = "Recent conversation:\n"
            for msg in recent:
                role = "Agent" if msg["role"] == "assistant" else "User"
                history_context += f"{role}: {msg['content']}\n"

        prompt = f"""You are an intent classifier for a voice assistant.
System Context (Use Case Info):
{_SYSTEM_PROMPT}

Current State of workflow: {current_state}
{history_context}
User just said: "{user_text}"

Understand user input in any language, but classify intent only.
Categories:
- CONFIRMATION: User says yes, done, completed, ho gaya, haan, hogya, kar liya, etc.
- REJECTION: User says no, not now, cancel, nahi, nhi, nhi hua, etc.
- QUERY: User asks a question, expresses a problem, or needs help.
- REPEAT: User asks to repeat (phir se, dobara, dubara, repeat).
- WAIT: User asks to wait (ruko, hold on, one minute).
- GOODBYE: User wants to end the call (bye, goodbye, dhanyavaad, shubh ho).
- EXIT: User wants to stop now and do it later (baad me, phir kabhi, not now, stop, exit).
- NOISE: Irrelevant noise or greetings.

Output ONLY the category name.
"""
        try:
            result = self._call_chat_backend(
                messages=[
                    {"role": "system", "content": "You are a precise intent classifier. Output only one word."},
                    {"role": "user", "content": prompt}
                ],
                model_name=self.intent_model if self.backend == "vllm" else self.ollama_model,
                temperature=0,
                max_tokens=10,
            ).strip().upper()
            for cat in ["CONFIRMATION", "REJECTION", "QUERY", "REPEAT", "WAIT", "GOODBYE", "EXIT"]:
                if cat in result:
                    print(f"[LLM-Classifier] Classified '{user_text}' as {cat}")
                    return cat
            return "NOISE"
        except Exception as e:
            print(f"[LLM-Classifier] Error: {e}")
            return "NOISE"

    def route_intent(self, user_text: str, current_state: str, history: list = None) -> dict:
        """
        Classify the user's intent into: knowledge_query, conversation_control, workflow, general_response.
        Returns a dict e.g. {"intent": "knowledge_query", "action": null, "topic": "domain specific topic"}
        """
        history_context = ""
        if history:
            recent = history[-4:]
            history_context = "Recent conversation:\n"
            for msg in recent:
                role = "Agent" if msg["role"] == "assistant" else "User"
                history_context += f"{role}: {msg['content']}\n"

        prompt = f"""You are a precise routing engine for a voice assistant.
System Context (Use Case Info):
{_SYSTEM_PROMPT}

Analyze the user's speech in the context of the current state and conversation history.

Current State of workflow: {current_state}
{history_context}
User just said: "{user_text}"

Classify into one of the following JSON format:
{{
  "intent": "knowledge_query" | "conversation_control" | "workflow" | "general_response",
  "action": "repeat" | "wait" | "continue" | "confirm" | "reject" | "exit" | null,
  "topic": "<extract the main topic being discussed as a string>" | null
}}

Intent Descriptions:
- knowledge_query: User is asking a question related to the business/domain (refer to System Context).
- conversation_control: User requests to repeat, wait/hold, stop, or exit.
- workflow: User is providing information or confirming/rejecting a question asked by the bot.
- general_response: Greetings, simple acknowledgment without state transition, or noise/greetings.

Output ONLY valid JSON.
"""
        try:
            result_raw = self._call_chat_backend(
                messages=[
                    {"role": "system", "content": "You are a precise intent routing engine. Output only JSON."},
                    {"role": "user", "content": prompt}
                ],
                model_name=self.intent_model if self.backend == "vllm" else self.ollama_model,
                temperature=0,
                max_tokens=60,
                response_format="json",
            ).strip()
            
            try:
                data = json.loads(result_raw)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", result_raw, flags=re.DOTALL)
                data = json.loads(match.group(0)) if match else {"intent": "workflow"}
                
            if not isinstance(data, dict):
                data = {"intent": "workflow"}
            
            # Normalize intent and action
            data["intent"] = str(data.get("intent") or "workflow").strip().lower()
            data["action"] = str(data.get("action") or "").strip().lower() if data.get("action") else None
            data["topic"] = str(data.get("topic") or "").strip().lower() if data.get("topic") else None
            print(f"[LLM-Router] Routed '{user_text}' -> {data}")
            return data
        except Exception as e:
            print(f"[LLM-Router] Error: {e}")
            return {"intent": "workflow"}

    def classify_slot(self, current_state: str, user_text: str, last_bot_message: str = "") -> dict:
        """
        Lightweight JSON slot classifier for FSM fallback.
        Schema is defined per-state in flow.json via 'llm_slot_schema'.
        """
        state = (current_state or "").strip().upper()
        utterance = (user_text or "").strip()
        if not utterance:
            return {"intent": "unclear"}

        prompt = f"""System Context:
{_SYSTEM_PROMPT}

State: {state}
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

    def generate_response(
        self,
        user_message: str,
        conversation_history: list,
        session_info: dict,
        context: str = "",
    ) -> str:
        """
        Generate a domain-scoped conversational reply.

        Guardrails enforced here:
        1. The system message leads with the use-case system_prompt so the LLM
           knows exactly what domain it is operating in.
        2. A hard OFF-TOPIC RULE is injected: if the user's question is unrelated
           to the domain, the LLM MUST return a polite, short refusal and redirect.
        3. Response is capped at 2-3 short sentences for voice delivery.
        4. Markdown, lists, and tables are forbidden (voice-unsafe).
        """
        # Build conversation history block (last 6 turns max)
        history_block = ""
        if conversation_history:
            for msg in conversation_history[-6:]:
                role = "Agent" if msg.get("role") == "assistant" else "User"
                history_block += f"{role}: {msg.get('content', '')}\n"

        # Session context
        session_block = ""
        if session_info:
            parts = []
            for k, v in session_info.items():
                if v:
                    parts.append(f"{k}: {v}")
            if parts:
                session_block = "Known info: " + " | ".join(parts)

        # RAG context
        context_block = f"\nRelevant knowledge:\n{context}\n" if context else ""

        system_message = f"""{_SYSTEM_PROMPT}

RESPONSE RULES — MUST FOLLOW:
1. Answer ONLY questions that fall within the domain described above.
2. If the user asks about ANYTHING off-topic (other subjects, general knowledge, unrelated questions), respond with ONE short, calm refusal sentence, then redirect to the domain. Example: "I can only help with [domain topic]. Is there anything about that I can assist you with?"
3. Never guess, fabricate, or use information not present in the knowledge base or conversation.
4. Keep your reply to 2-3 short sentences only — this is a voice call, not a text chat.
5. Do NOT use markdown, bullet points, numbered lists, or tables. Plain spoken sentences only.
6. Do NOT repeat the user's question back to them.
7. Speak in the same language the user used (English, Hindi, or Hinglish).
{context_block}
{session_block}"""

        messages = [
            {"role": "system", "content": system_message},
        ]

        # Inject conversation history
        if history_block:
            messages.append({
                "role": "user",
                "content": f"Conversation so far:\n{history_block}\nUser's latest message: {user_message}"
            })
        else:
            messages.append({"role": "user", "content": user_message})

        try:
            response = self._call_chat_backend(
                messages=messages,
                model_name=self.vllm_model if self.backend == "vllm" else self.ollama_model,
                temperature=0.3,
                max_tokens=120,
            ).strip()
            print(f"[LLM-Response] Generated: '{response[:80]}...'")
            return response
        except Exception as e:
            print(f"[LLM-Response] Error: {e}")
            return ""

    def is_available(self) -> bool:

        return True


# ── Singleton ─────────────────────────────────────────────────────────────────
_llm: Optional[LLMEngine] = None


def get_llm() -> LLMEngine:
    global _llm
    if _llm is None:
        _llm = LLMEngine()
    return _llm

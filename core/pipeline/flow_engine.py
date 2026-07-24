"""
flow_engine.py — Generic JSON-driven conversation state machine.
================================================================
The entire conversation flow is defined in flow.json per use case.
This engine reads that JSON and executes state transitions.
No hardcoded ConversationState enums. No use-case-specific logic here.

flow.json structure:
{
  "initial_state": "CALL_CONNECTED",
  "llm_fallback_states": ["STATE_A", "STATE_B"],
  "states": {
    "CALL_CONNECTED": {
      "script": {"en": "Hello...", "hi": "Namaste..."},
      "script_key": "step_script_key",         # optional: load from KB
      "transitions": {
        "affirm":  "NEXT_STATE",
        "deny":    "CLOSING",
        "unknown": "CALL_CONNECTED"
      },
      "llm_fallback": false,
      "terminal": false,
      "slots_to_capture": ["field_name"],
      "action": "action_name"                  # optional: call hooks.py
    }
  }
}
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Any


# ── Session context ───────────────────────────────────────────────────────────

@dataclass
class SessionContext:
    """Mutable state for one call session."""
    session_id:           str
    current_state:        str = ""
    previous_state:       str = ""
    language:             str = "en"
    turn_count:           int = 0
    last_bot_text:        str = ""
    call_ended:           bool = False
    pending_audio_clips:  list = field(default_factory=list)
    metadata:             dict = field(default_factory=dict)


# ── Flow Engine ───────────────────────────────────────────────────────────────

class FlowEngine:
    """
    Generic state machine that executes any flow.json definition.

    Usage:
        engine = FlowEngine(flow_dict, retriever, llm, hooks_module)
        response, is_terminal = engine.process(session, intent_name, intent_data, memory)
    """

    def __init__(
        self,
        flow: dict,
        retriever=None,
        llm=None,
        hooks=None,
    ):
        self.flow      = flow
        self.retriever = retriever
        self.llm       = llm
        self.hooks     = hooks

        self.initial_state     = flow.get("initial_state", "")
        self.states            = flow.get("states", {})
        self.llm_fallback_set  = set(flow.get("llm_fallback_states", []))

        print(
            f"[FlowEngine] Loaded {len(self.states)} states, "
            f"initial={self.initial_state}, "
            f"llm_fallback_states={len(self.llm_fallback_set)}"
        )

    def init_session(self, session: SessionContext) -> str:
        """Set the initial state and return the opening script."""
        session.current_state = self.initial_state
        return self._get_script(self.initial_state, session.language)

    def process(
        self,
        session: SessionContext,
        intent_name: str,
        intent_data: dict,
        memory=None,
    ) -> tuple[str, bool]:
        """
        Handle one user turn.

        Returns:
            (response_text, is_terminal)
        """
        state_key = session.current_state
        state_def = self.states.get(state_key, {})
        session.turn_count += 1

        # ── 1. Check for optional state-level action (hooks) ──────────────────
        action = state_def.get("action")
        if action and self.hooks and hasattr(self.hooks, action):
            try:
                hook_result = getattr(self.hooks, action)(
                    session=session,
                    intent_name=intent_name,
                    intent_data=intent_data,
                    memory=memory,
                )
                # Hook can return (text, next_state) to take full control
                if isinstance(hook_result, tuple) and len(hook_result) == 2:
                    text, next_state = hook_result
                    if next_state:
                        self._transition(session, next_state)
                    return text, self._is_terminal(session.current_state)
            except Exception as e:
                print(f"[FlowEngine] Hook '{action}' error: {e}")

        # ── 2. Determine next state from transition table ──────────────────────
        transitions = state_def.get("transitions", {})

        # Map intent names to transitions — try exact match, then prefix match
        next_state = (
            transitions.get(intent_name)
            or transitions.get(intent_name.split("_")[0])
            or transitions.get("default")
        )

        # ── 3. LLM fallback for states that need it ────────────────────────────
        if (
            next_state is None
            and state_def.get("llm_fallback", False)
            and self.llm
            and memory is not None
        ):
            next_state = self._llm_transition(
                state_key, intent_name, memory, transitions, session
            )

        # If still no transition, stay in current state
        if next_state is None:
            next_state = state_key

        # ── 4. Transition ──────────────────────────────────────────────────────
        self._transition(session, next_state)

        # ── 5. Build response ─────────────────────────────────────────────────
        new_state_def = self.states.get(next_state, {})
        is_terminal   = new_state_def.get("terminal", False)

        # Try RAG for topic queries
        if intent_name == "topic_query" and intent_data.get("topic") and self.retriever:
            rag_answer = self.retriever.retrieve(intent_data["topic"])
            if rag_answer:
                return rag_answer, is_terminal

        # LLM for QUERY intents in LLM-enabled states
        if (
            intent_name in {"unknown", "topic_query"}
            and next_state in self.llm_fallback_set
            and self.llm
            and memory is not None
        ):
            try:
                rag_context = ""
                if self.retriever:
                    rag_context = self.retriever.retrieve(
                        memory.history[-1]["content"] if memory.history else ""
                    ) or ""
                llm_response = self.llm.generate_response(
                    user_message=memory.history[-1]["content"] if memory.history else "",
                    conversation_history=memory.get_recent_history(),
                    session_info=memory.get_session_info(),
                    context=rag_context,
                )
                session.last_bot_text = llm_response
                return llm_response, is_terminal
            except Exception as e:
                print(f"[FlowEngine] LLM response error: {e}")

        # Default: return new state's script
        script = self._get_script(next_state, session.language)
        script = self._format_script(script, session, memory)
        session.last_bot_text = script
        return script, is_terminal

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _transition(self, session: SessionContext, next_state: str) -> None:
        if next_state not in self.states and next_state != session.current_state:
            print(f"[FlowEngine] WARNING: unknown state '{next_state}', staying in '{session.current_state}'")
            return
        session.previous_state = session.current_state
        session.current_state  = next_state
        print(f"[FlowEngine] {session.previous_state} → {next_state} | session={session.session_id[:8]}")

    def _get_script(self, state_key: str, language: str = "en") -> str:
        state_def = self.states.get(state_key, {})

        # Inline script dict
        script = state_def.get("script")
        if isinstance(script, dict):
            return script.get(language) or script.get("en") or ""
        if isinstance(script, str):
            return script

        # KB lookup via retriever
        script_key = state_def.get("script_key")
        if script_key and self.retriever:
            try:
                kb_script = self.retriever.get_step_script(script_key)
                if kb_script:
                    return kb_script
            except Exception:
                pass

        return ""

    def _format_script(self, script: str, session: SessionContext, memory=None) -> str:
        """Replace {placeholders} in scripts with values from session/memory."""
        if not script or "{" not in script:
            return script
        ctx: dict = {"language": session.language}
        if memory:
            ctx.update(memory.get_session_info())
            ctx.update(memory.lead.as_dict())
        try:
            return script.format_map(_SafeDict(ctx))
        except Exception:
            return script

    def _llm_transition(
        self,
        state_key: str,
        intent_name: str,
        memory,
        transitions: dict,
        session: SessionContext,
    ) -> Optional[str]:
        """
        Ask the LLM to classify the intent more precisely and map to a transition.
        Returns the next state key, or None.
        """
        if not memory or not memory.history:
            return None
        last_user = memory.history[-1]["content"] if memory.history else ""
        try:
            slot_result = self.llm.classify_slot(
                current_state=state_key,
                user_text=last_user,
                last_bot_message=session.last_bot_text,
            )
            llm_intent = slot_result.get("intent", "unclear")
            # Try to map LLM intent to a transition
            mapped = (
                transitions.get(llm_intent)
                or transitions.get(llm_intent.split("_")[0])
                or transitions.get("default")
            )
            if mapped:
                print(f"[FlowEngine] LLM mapped '{llm_intent}' → '{mapped}' in state {state_key}")
                return mapped
        except Exception as e:
            print(f"[FlowEngine] LLM transition error: {e}")
        return None

    def _is_terminal(self, state_key: str) -> bool:
        return self.states.get(state_key, {}).get("terminal", False)

    def get_state_def(self, state_key: str) -> dict:
        return self.states.get(state_key, {})

    def is_llm_fallback_state(self, state_key: str) -> bool:
        return state_key in self.llm_fallback_set


class _SafeDict(dict):
    """dict subclass that returns '{key}' for missing keys (safe format_map)."""
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"

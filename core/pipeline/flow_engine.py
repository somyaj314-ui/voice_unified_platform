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


# ── Barge-In Configuration ────────────────────────────────────────────────────
# ENABLE_BARGE_IN: Master switch. Set False to restore old echo-suppression.
ENABLE_BARGE_IN: bool = True

# User must produce this many milliseconds of *consecutive* speech frames while
# the bot is playing before barge-in fires. Avoids triggering on breath/static.
BARGE_IN_MIN_SPEECH_MS: int = 400

# How many consecutive Silero-positive frames (each ~32ms) are needed to confirm
# real speech. 3 × 32ms = ~96ms of consecutive frames → validated rapidly.
CONSECUTIVE_SPEECH_FRAMES_FOR_INTERRUPT: int = 3



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

    # ── Barge-in state ────────────────────────────────────────────────────────
    # True for the single turn immediately following a barge-in event.
    # Flow engine can use this to craft a shorter "I heard you" acknowledgement.
    was_barge_in:         bool = False

    # True while bot playback is being interrupted but user utterance has not
    # yet been fully processed. Cleared when process() is called after barge-in.
    interruption_active:  bool = False

    # Stack of state keys pushed when an interruption fires.
    # Allows the engine to know what topic was being discussed before the user
    # cut in — useful for "resume after interruption" logic.
    topic_stack:          list = field(default_factory=list)



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

        if "workflows" in flow:
            # Legacy format support (e.g. zipnach)
            wf = list(flow["workflows"].values())[0]
            self.initial_state = wf.get("initial_state", "")
            self.states = {}
            for s in wf.get("states", []):
                state_id = s.get("state_id")
                if state_id:
                    self.states[state_id] = s
            self.llm_fallback_set = set()
        else:
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

        # ── 0. Clear barge-in one-shot flags ─────────────────────────────────
        # was_barge_in is True only for this turn (the turn right after barge-in).
        # interruption_active is cleared now that the user utterance is being
        # processed — bot is no longer being interrupted.
        prev_was_barge_in       = session.was_barge_in
        session.was_barge_in    = False
        session.interruption_active = False


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

        # ── 5b. Off-topic / unknown intent guardrail ──────────────────────────
        # If the intent is UNKNOWN and we are NOT in an LLM-fallback state
        # (so there's no LLM handler above to catch it), return a polite domain
        # refusal instead of silently re-reading the current state script.
        # This prevents the bot from awkwardly repeating itself on irrelevant input.
        if intent_name == "unknown" and next_state not in self.llm_fallback_set:
            refusal = self._off_topic_refusal(session.language)
            print(f"[FlowEngine] Off-topic refusal triggered | state={state_key} | lang={session.language}")
            session.last_bot_text = refusal
            # Stay in current state — don't advance the flow
            return refusal, False

        # Default: return new state's script
        script = self._get_script(next_state, session.language)
        script = self._format_script(script, session, memory)
        session.last_bot_text = script
        return script, is_terminal


    # ── Barge-in API ──────────────────────────────────────────────────────────

    def handle_barge_in(self, session: SessionContext) -> None:
        """
        Called by server.py the moment user speech is detected during bot playback.

        - Pushes the current conversation state onto the topic_stack so the engine
          can know what was being discussed before the interruption (for resumption).
        - Marks interruption_active=True and was_barge_in=True so the next call to
          process() knows this turn is a post-interruption turn.
        """
        if not ENABLE_BARGE_IN:
            return

        # Only push state if it's not already at the top of the stack
        if not session.topic_stack or session.topic_stack[-1] != session.current_state:
            session.topic_stack.append(session.current_state)

        session.interruption_active = True
        session.was_barge_in        = True

        print(
            f"[FlowEngine] Barge-in recorded | state={session.current_state} | "
            f"topic_stack={session.topic_stack} | session={session.session_id[:8]}"
        )

    def stop_user_speaking_barge_in(self, session: SessionContext) -> None:
        """
        Called when the user finishes speaking after a barge-in and processing
        is complete. Resets interruption_active (was_barge_in already cleared
        by process()). The topic_stack is left intact for higher-level logic.
        """
        session.interruption_active = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _off_topic_refusal(self, language: str = "en") -> str:
        """
        Return a short, calm, voice-friendly off-topic refusal line.
        Picks the right language variant from the knowledge base (script.json).
        These are hardcoded fallbacks — the LLM's generate_response() handles
        more nuanced refusals when the LLM is available.
        """
        lang = (language or "en").strip().lower()

        # 1. Try to fetch use-case specific refusal from the KB
        if self.retriever:
            try:
                kb_script = self.retriever.get_step_script(f"off_topic_refusal_{lang}")
                if kb_script:
                    return kb_script
            except Exception:
                pass

        # 2. Hardcoded generic fallbacks if not found in KB
        refusals = {
            "en": (
                "I can only help with queries related to this service. "
                "Is there anything else I can assist you with?"
            ),
            "hi": (
                "Main sirf is service se related sawaalon mein aapki madad kar sakta hoon. "
                "Kya koi aur sawaal hai jo main aapke liye answer kar sakta hoon?"
            ),
            "hinglish": (
                "Main sirf is service ke baare mein help kar sakta hoon. "
                "Koi aur query hai kya?"
            ),
        }
        return refusals.get(lang, refusals["en"])

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
            
        # Legacy format fallback (e.g. zipnach)
        legacy_script = state_def.get(f"bot_says_{language}")
        if legacy_script:
            return legacy_script
        legacy_en = state_def.get("bot_says_en")
        if legacy_en:
            return legacy_en

        # KB lookup via retriever
        script_key = state_def.get("script_key")
        if script_key and self.retriever:
            try:
                kb_script = self.retriever.get_step_script(script_key)
                if kb_script:
                    if isinstance(kb_script, dict):
                        return kb_script.get(language) or kb_script.get("en") or ""
                    return str(kb_script)
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

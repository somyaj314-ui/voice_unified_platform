"""
voice_pipeline.py — Main orchestrator for voice sessions with multi-voice support
"""

from __future__ import annotations

import time
import traceback
from typing import Optional

import numpy as np

from core.pipeline.call_context import CallContext
from core.pipeline.debug_logger import get_debug_logger
from core.pipeline.flow_engine import FlowEngine, SessionContext
from core.pipeline.intent_engine import classify_intent
from core.pipeline.llm_engine import get_llm
from core.pipeline.memory import memory_manager
from core.pipeline.rag_knowledge import get_retriever
from core.pipeline.stt_engine import get_stt
from core.pipeline.tts_pipeline import tts_pipeline


class FlowSession:
    """Tracks session runtime data."""

    def __init__(self, session_id: str, context: CallContext, flow_engine: FlowEngine):
        self.session_id = session_id
        self.context = context
        self.flow_engine = flow_engine
        self.session_ctx = SessionContext(session_id=session_id)

        self.last_audio_time = time.monotonic()
        self.last_transcription = ""
        self.transcription_count = 0
        self.last_transcript_time = 0.0

        memory = memory_manager.get_or_create(session_id)
        if context.customer_name:
            memory.lead.set("customer_name", context.customer_name)
        if context.lead_id:
            memory.lead.set("lead_id", context.lead_id)

    @property
    def is_active(self) -> bool:
        return not self.session_ctx.call_ended


class VoicePipeline:
    """Main orchestrator for voice sessions."""

    def __init__(self):
        self._flows: dict[str, FlowSession] = {}

    def create_session(
        self,
        session_id: str,
        context: CallContext,
        flow_config: dict,
        hooks_module=None
    ) -> FlowSession:
        """Initialize a new call session with its specific flow definition."""
        flow_engine = FlowEngine(
            flow=flow_config,
            retriever=get_retriever(),
            llm=get_llm(),
            hooks=hooks_module,
        )

        flow = FlowSession(session_id, context, flow_engine)
        self._flows[session_id] = flow
        return flow

    def get_session(self, session_id: str) -> Optional[FlowSession]:
        return self._flows.get(session_id)

    def remove_session(self, session_id: str) -> None:
        self._flows.pop(session_id, None)
        memory_manager.remove(session_id)

    def start_call(self, session_id: str) -> str:
        """Trigger the opening sequence of the call."""
        flow = self.get_session(session_id)
        if not flow:
            return ""

        script = flow.flow_engine.init_session(flow.session_ctx)

        debug = get_debug_logger()
        debug.log_call_start(session_id, flow.context.as_dict())
        debug.log_greeting(session_id, script, flow.context.as_dict())

        return script

    def process_utterance(self, session_id: str, utterance: str) -> tuple[str, bool]:
        """
        Process a finalized text utterance through the NLU pipeline.
        Returns: (bot_response_text, is_terminal)
        """
        flow = self.get_session(session_id)
        if not flow or not flow.is_active:
            return "", True

        memory = memory_manager.get(session_id)
        if not memory:
            memory = memory_manager.get_or_create(session_id)

        memory.update_entities(utterance)

        intents = classify_intent(utterance, current_state=flow.session_ctx.current_state)
        primary = intents[0]

        bot_reply, is_terminal = flow.flow_engine.process(
            session=flow.session_ctx,
            intent_name=primary.name,
            intent_data=primary.data,
            memory=memory,
        )

        if is_terminal:
            flow.session_ctx.call_ended = True

        memory.add_turn(utterance, bot_reply)
        get_debug_logger().log_turn(
            session_id=session_id,
            user_text=utterance,
            bot_response=bot_reply,
            flow_state=flow.session_ctx.current_state,
            is_terminal=is_terminal,
            metadata={"intent": primary.name, "confidence": primary.confidence},
        )

        return bot_reply, is_terminal

    def process_audio(self, session_id: str, audio_np: np.ndarray, sample_rate: int = 16000) -> tuple[str, bool]:
        """
        Process a finalized audio segment through STT, then NLU.
        Returns: (bot_response_text, is_terminal)
        """
        flow = self.get_session(session_id)
        if not flow:
            return "", True

        flow.last_audio_time = time.monotonic()

        stt = get_stt()
        try:
            transcript = stt.transcribe(audio_np, sample_rate)
        except Exception as e:
            print(f"[Pipeline] STT error: {e}")
            traceback.print_exc()
            return "", False

        if not transcript.strip():
            return "", False

        return self.process_utterance(session_id, transcript)

    def synthesize_text(
        self,
        text: str,
        session_id: str,
        language: str = "en",
        voice: str | None = None
    ) -> bytes:
        """
        Synthesize text to raw mulaw bytes via the configured TTS pipeline with multi-voice support.
        """
        voice_id = voice
        if not voice_id:
            flow = self.get_session(session_id)
            if flow and flow.context:
                voice_id = getattr(flow.context, "voice", None)

        return tts_pipeline.generate_audio(text, language, voice=voice_id)


# Singleton
voice_pipeline = VoicePipeline()

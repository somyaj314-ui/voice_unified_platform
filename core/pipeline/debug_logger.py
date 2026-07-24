"""
debug_logger.py — Call conversation debug logger.
Generic — logs every user transcription, bot response, and call metadata.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Any


class CallDebugLogger:
    """Logs all conversational turns in JSON format for debugging."""

    def __init__(self, debug_dir: str = "debug_logs"):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(exist_ok=True)
        self.calls_dir = self.debug_dir / "calls"
        self.calls_dir.mkdir(exist_ok=True)
        self.index_file = self.debug_dir / "calls_index.jsonl"

    def _ensure_session_file(self, session_id: str) -> Path:
        return self.calls_dir / f"{session_id}.jsonl"

    def log_turn(
        self,
        session_id: str,
        user_text: str,
        bot_response: str,
        call_context: Optional[dict] = None,
        flow_state: Optional[str] = None,
        language: Optional[str] = None,
        is_terminal: bool = False,
        metadata: Optional[dict] = None,
    ) -> None:
        timestamp = datetime.utcnow().isoformat() + "Z"
        turn_log = {
            "timestamp": timestamp,
            "session_id": session_id,
            "user_text": user_text,
            "bot_response": bot_response,
            "flow_state": flow_state,
            "language": language,
            "is_terminal": is_terminal,
        }
        if call_context:
            turn_log["call_context"] = call_context
        if metadata:
            turn_log["metadata"] = metadata

        session_file = self._ensure_session_file(session_id)
        with open(session_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn_log) + "\n")

        index_entry = {
            "timestamp": timestamp,
            "session_id": session_id,
            "is_terminal": is_terminal,
            "user_text_preview": user_text[:50] if user_text else "",
            "bot_response_preview": bot_response[:50] if bot_response else "",
            "flow_state": flow_state,
        }
        with open(self.index_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(index_entry) + "\n")

    def log_call_start(
        self,
        session_id: str,
        call_context: Optional[dict] = None,
    ) -> None:
        timestamp = datetime.utcnow().isoformat() + "Z"
        call_start = {
            "timestamp": timestamp,
            "session_id": session_id,
            "event": "call_start",
        }
        if call_context:
            call_start["call_context"] = call_context
        session_file = self._ensure_session_file(session_id)
        with open(session_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(call_start) + "\n")

    def log_call_end(
        self,
        session_id: str,
        reason: Optional[str] = None,
        total_turns: int = 0,
    ) -> None:
        timestamp = datetime.utcnow().isoformat() + "Z"
        call_end = {
            "timestamp": timestamp,
            "session_id": session_id,
            "event": "call_end",
            "reason": reason,
            "total_turns": total_turns,
        }
        session_file = self._ensure_session_file(session_id)
        with open(session_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(call_end) + "\n")

    def log_greeting(
        self,
        session_id: str,
        bot_greeting: str,
        call_context: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        timestamp = datetime.utcnow().isoformat() + "Z"
        greeting_log = {
            "timestamp": timestamp,
            "session_id": session_id,
            "event": "greeting",
            "bot_greeting": bot_greeting,
        }
        if call_context:
            greeting_log["call_context"] = call_context
        if metadata:
            greeting_log["metadata"] = metadata
        session_file = self._ensure_session_file(session_id)
        with open(session_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(greeting_log) + "\n")

    def get_session_log(self, session_id: str) -> list[dict]:
        session_file = self.calls_dir / f"{session_id}.jsonl"
        turns = []
        if session_file.exists():
            with open(session_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        turns.append(json.loads(line))
        return turns

    def get_session_summary(self, session_id: str) -> dict:
        turns = self.get_session_log(session_id)
        if not turns:
            return {}
        user_turns = [t for t in turns if "user_text" in t]
        bot_turns = [t for t in turns if "bot_response" in t or "bot_greeting" in t]
        summary = {
            "session_id": session_id,
            "start_time": turns[0].get("timestamp"),
            "end_time": turns[-1].get("timestamp"),
            "total_events": len(turns),
            "user_turns": len(user_turns),
            "bot_turns": len(bot_turns),
            "conversation": [],
        }
        for turn in turns:
            if turn.get("event") == "call_start":
                summary["conversation"].append({
                    "type": "call_start",
                    "timestamp": turn.get("timestamp"),
                    "context": turn.get("call_context"),
                })
            elif turn.get("event") == "greeting":
                summary["conversation"].append({
                    "type": "greeting",
                    "timestamp": turn.get("timestamp"),
                    "bot_text": turn.get("bot_greeting"),
                })
            elif "user_text" in turn:
                summary["conversation"].append({
                    "type": "turn",
                    "timestamp": turn.get("timestamp"),
                    "user_text": turn.get("user_text"),
                    "bot_response": turn.get("bot_response"),
                    "flow_state": turn.get("flow_state"),
                    "language": turn.get("language"),
                })
            elif turn.get("event") == "call_end":
                summary["conversation"].append({
                    "type": "call_end",
                    "timestamp": turn.get("timestamp"),
                    "reason": turn.get("reason"),
                })
        return summary


# Singleton
_debug_logger: Optional[CallDebugLogger] = None


def get_debug_logger() -> CallDebugLogger:
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = CallDebugLogger()
    return _debug_logger

"""
profiler.py — Lightweight wall-clock latency profiler for the voice pipeline.

Measures what users actually feel: the gap between their last word
and the first byte of the bot's response.

Usage:
    from pipeline.profiler import TurnProfiler

    prof = TurnProfiler(session_id)
    prof.mark("asr_start")
    # ... ASR runs ...
    prof.mark("asr_end")
    prof.mark("flow_end")
    prof.mark("tts_end")
    prof.mark("first_byte_sent")
    prof.finish()          # prints + logs the turn summary
"""

from __future__ import annotations

import time
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Where to write the profile log ───────────────────────────────────────────
_LOG_DIR = Path(__file__).parent.parent / "debug_logs" / "profiler"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Stage name constants (use these so typos are caught at import time) ──────
class Stage:
    SEGMENT_RECEIVED  = "segment_received"   # VAD finished, segment handed off
    ASR_START         = "asr_start"
    ASR_END           = "asr_end"            # transcription ready
    FLOW_START        = "flow_start"
    FLOW_END          = "flow_end"           # response_text ready
    RAG_START         = "rag_start"          # optional
    RAG_END           = "rag_end"            # optional
    LLM_START         = "llm_start"          # optional
    LLM_END           = "llm_end"            # optional
    TTS_START         = "tts_start"
    TTS_END           = "tts_end"            # audio bytes ready
    FIRST_BYTE_SENT   = "first_byte_sent"    # first chunk enqueued
    PLAYBACK_DONE     = "playback_done"      # mark event received


@dataclass
class TurnProfiler:
    """
    One instance per user turn.  Call mark() at each pipeline stage,
    then finish() to print and persist the results.
    """
    session_id: str
    turn_number: int = 0

    _marks: dict[str, float] = field(default_factory=dict)
    _start: float = field(default_factory=time.monotonic)

    # Metadata attached at finish time
    transcript:    str = ""
    response_text: str = ""
    flow_state:    str = ""
    used_rag:      bool = False
    used_llm:      bool = False
    is_clip:       bool = False   # True when response was a pre-recorded clip

    def mark(self, stage: str) -> float:
        """Record the current wall-clock time for a named stage."""
        t = time.monotonic()
        self._marks[stage] = t
        return t

    def elapsed(self, from_stage: str, to_stage: str) -> Optional[float]:
        """
        Return the time in milliseconds between two named stages.
        Returns None if either stage was not recorded.
        """
        t0 = self._marks.get(from_stage)
        t1 = self._marks.get(to_stage)
        if t0 is None or t1 is None:
            return None
        return (t1 - t0) * 1000.0

    def total_latency_ms(self) -> Optional[float]:
        """
        The latency the user feels: from segment_received to first_byte_sent.
        This is the single most important number.
        """
        return self.elapsed(Stage.SEGMENT_RECEIVED, Stage.FIRST_BYTE_SENT)

    def finish(self) -> dict:
        """
        Build the summary, print it, write it to the profile log, and return it.
        """
        summary = self._build_summary()
        self._print_summary(summary)
        self._write_log(summary)
        return summary

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_summary(self) -> dict:
        def ms(a, b):
            v = self.elapsed(a, b)
            return round(v, 1) if v is not None else None

        total = self.total_latency_ms()

        return {
            "session_id":        self.session_id,
            "turn":              self.turn_number,
            "transcript":        self.transcript[:120],
            "response_preview":  self.response_text[:80],
            "flow_state":        self.flow_state,
            "used_rag":          self.used_rag,
            "used_llm":          self.used_llm,
            "is_clip":           self.is_clip,
            # ── Segment latencies (ms) ──
            "asr_ms":            ms(Stage.ASR_START,         Stage.ASR_END),
            "flow_ms":           ms(Stage.FLOW_START,        Stage.FLOW_END),
            "rag_ms":            ms(Stage.RAG_START,         Stage.RAG_END),
            "llm_ms":            ms(Stage.LLM_START,         Stage.LLM_END),
            "tts_ms":            ms(Stage.TTS_START,         Stage.TTS_END),
            "playback_ms":       ms(Stage.FIRST_BYTE_SENT,   Stage.PLAYBACK_DONE),
            # ── Key totals ──
            "total_latency_ms":  round(total, 1) if total is not None else None,
            "segment_to_asr_ms": ms(Stage.SEGMENT_RECEIVED, Stage.ASR_START),
            "tts_to_first_byte_ms": ms(Stage.TTS_END,       Stage.FIRST_BYTE_SENT),
            # Internal: used by record_playback_done to compute actual playback duration
            "_tts_end_abs":      self._marks.get(Stage.TTS_END),
        }

    def _print_summary(self, s: dict) -> None:
        total = s["total_latency_ms"]
        flag  = " ⚠ SLOW" if total and total > 4000 else ""

        print(
            f"[PROFILER] turn={s['turn']} session={s['session_id'][:8]}"
            f" | asr={s['asr_ms']}ms"
            f" flow={s['flow_ms']}ms"
            f" {'rag=' + str(s['rag_ms']) + 'ms ' if s['used_rag'] else ''}"
            f"{'llm=' + str(s['llm_ms']) + 'ms ' if s['used_llm'] else ''}"
            f"tts={s['tts_ms']}ms"
            f" | TOTAL={total}ms{flag}"
            f" | state={s['flow_state']}"
        )

    def _write_log(self, summary: dict) -> None:
        try:
            log_path = _LOG_DIR / f"{self.session_id}.jsonl"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary) + "\n")
        except Exception as e:
            print(f"[PROFILER] Failed to write log: {e}")


# ── Session-level aggregator ──────────────────────────────────────────────────

class SessionProfiler:
    """
    Tracks all turns in a session and prints a summary on call end.

    Usage:
        sp = SessionProfiler(session_id)
        prof = sp.new_turn()          # creates a TurnProfiler
        ...
        sp.finish()                   # prints per-turn + session averages
    """

    def __init__(self, session_id: str):
        self.session_id  = session_id
        self._turns: list[dict] = []
        self._current: Optional[TurnProfiler] = None

    def new_turn(self) -> TurnProfiler:
        # If there's an unfinished turn, finish it before starting a new one
        # to prevent orphaned profiler objects from accumulating.
        if self._current is not None:
            self._turns.append(self._current.finish())
        self._current = TurnProfiler(
            session_id=self.session_id,
            turn_number=len(self._turns) + 1,
        )
        return self._current

    def record(self, summary: dict) -> None:
        """Called by TurnProfiler.finish() implicitly via finish_turn()."""
        self._turns.append(summary)

    def finish_turn(self) -> dict:
        """Finish the current turn and record it."""
        if self._current is None:
            return {}
        summary = self._current.finish()
        self._turns.append(summary)
        self._current = None
        return summary

    def record_playback_done(self) -> None:
        """
        Called when SmartFlo mark event arrives — playback is confirmed complete.
        Updates the last finished turn with the actual playback duration.
        This is called AFTER finish_turn(), so we patch the stored dict directly.
        """
        if not self._turns:
            return
        last = self._turns[-1]
        # Calculate playback_ms from first_byte to now
        # We store tts_end absolute time so we can compute it
        tts_end_abs = last.get("_tts_end_abs")
        if tts_end_abs is not None:
            playback_ms = (time.monotonic() - tts_end_abs) * 1000
            last["playback_ms"] = round(playback_ms, 1)
            print(
                f"[PROFILER] playback_done"
                f" turn={last.get('turn')}"
                f" playback_ms={last['playback_ms']:.0f}ms"
                f" session={self.session_id[:8]}"
            )

    def finish(self) -> None:
        """Print session-level latency summary."""
        if not self._turns:
            return

        totals = [t["total_latency_ms"] for t in self._turns if t.get("total_latency_ms")]
        asr    = [t["asr_ms"]           for t in self._turns if t.get("asr_ms")]
        tts    = [t["tts_ms"]           for t in self._turns if t.get("tts_ms")]

        def avg(lst):
            return round(sum(lst) / len(lst), 1) if lst else None

        def p95(lst):
            if not lst:
                return None
            s = sorted(lst)
            idx = max(0, int(len(s) * 0.95) - 1)
            return round(s[idx], 1)

        print(
            f"\n[SESSION_PROFILE] session={self.session_id[:8]}"
            f" turns={len(self._turns)}"
            f" | avg_latency={avg(totals)}ms  p95={p95(totals)}ms"
            f" | avg_asr={avg(asr)}ms  avg_tts={avg(tts)}ms"
        )

        # Write session summary
        try:
            log_path = _LOG_DIR / f"{self.session_id}_summary.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump({
                    "session_id": self.session_id,
                    "turns":      len(self._turns),
                    "avg_latency_ms": avg(totals),
                    "p95_latency_ms": p95(totals),
                    "avg_asr_ms":     avg(asr),
                    "avg_tts_ms":     avg(tts),
                    "turn_details":   self._turns,
                }, f, indent=2)
        except Exception as e:
            print(f"[PROFILER] Failed to write session summary: {e}")

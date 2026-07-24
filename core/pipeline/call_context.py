"""
call_context.py — Call/session metadata container.
Generic — no use-case-specific fields.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.parse import parse_qs, urlparse


@dataclass
class CallContext:
    session_id: str
    call_direction: str = "outbound"
    caller_id: str | None = None
    callee_id: str | None = None
    lead_id: str | None = None
    campaign_id: str | None = None
    attempt_number: int = 1
    call_reason: str | None = None
    current_workflow_id: str | None = None
    current_state_id: str | None = None
    preferred_language: str | None = None
    customer_name: str | None = None

    @classmethod
    def from_ws_path(cls, path: str, session_id: str) -> "CallContext":
        parsed = urlparse(path)
        query = parse_qs(parsed.query)

        def first(name: str) -> str | None:
            values = query.get(name)
            if not values:
                return None
            value = values[0].strip()
            return value or None

        attempt_number = 1
        raw_attempt = first("attempt_number")
        if raw_attempt:
            try:
                attempt_number = max(1, int(raw_attempt))
            except ValueError:
                attempt_number = 1

        return cls(
            session_id=session_id,
            call_direction=(first("call_direction") or "outbound").lower(),
            caller_id=first("caller_id"),
            callee_id=first("callee_id"),
            lead_id=first("lead_id"),
            campaign_id=first("campaign_id"),
            attempt_number=attempt_number,
            call_reason=first("call_reason"),
            current_workflow_id=first("workflow_id"),
            current_state_id=first("state_id"),
            preferred_language=first("language"),
            customer_name=first("customer_name"),
        )

    def as_dict(self) -> dict:
        return asdict(self)

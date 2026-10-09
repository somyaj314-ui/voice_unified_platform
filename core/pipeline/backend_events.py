"""
backend_events.py — WebSocket Broadcasting for Active Call Sessions
===================================================================
Provides backend_manager for broadcasting real-time call events (transcripts,
call statuses, barge-in alerts) to connected UI monitoring clients.
"""

import asyncio
import json
from fastapi import WebSocket


class BackendManager:
    def __init__(self):
        self._connections: dict[str, set[WebSocket]] = {}
        self._main_loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self._connections:
            self._connections[session_id] = set()
        self._connections[session_id].add(websocket)

    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self._connections:
            self._connections[session_id].discard(websocket)
            if not self._connections[session_id]:
                self._connections.pop(session_id, None)

    async def broadcast(self, session_id: str, event_type: str, data: dict):
        if session_id in self._connections:
            msg = json.dumps({"event": event_type, "data": data})
            for ws in list(self._connections[session_id]):
                try:
                    await ws.send_text(msg)
                except Exception:
                    self.disconnect(ws, session_id)

    def broadcast_sync(self, session_id: str, event_type: str, data: dict):
        if self._main_loop and self._main_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.broadcast(session_id, event_type, data), self._main_loop
            )


backend_manager = BackendManager()

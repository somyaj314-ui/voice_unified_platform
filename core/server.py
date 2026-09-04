"""
server.py — Generic WebRTC/WebSocket Server for Asterisk ARI.
=============================================================
Loads the USE_CASE configuration at startup.
Initializes all generic core components with use-case specific parameters.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import sys
import time
import uuid
import wave
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
import numpy as np
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Core platform imports ────────────────────────────────────────────────────
from core.loader import load_use_case
from core.pipeline.call_context import CallContext
from core.pipeline.debug_logger import get_debug_logger
from core.pipeline.flow_engine import FlowEngine, SessionContext
from core.pipeline.intent_engine import configure_intent
from core.pipeline.llm_engine import configure_llm
from core.pipeline.memory import configure_memory, memory_manager
from core.pipeline.rag_knowledge import configure_rag
from core.pipeline.stt_engine import configure_stt
from core.pipeline.tts_engine import configure_tts
from core.pipeline.voice_pipeline import voice_pipeline
from core.pipeline.tts_pipeline import tts_pipeline
from core.audio_utils import (
    StreamingSpeechSegmenter,
    float32_16k_to_mulaw_8k,
    mulaw_8k_to_float32_16k,
    compute_rms,
)

# ── Load Use Case Configuration ──────────────────────────────────────────────
_cfg = load_use_case()

# Initialize core modules with config
configure_llm(
    system_prompt=_cfg.get_system_prompt(),
    backend=_cfg.llm_backend,
    base_url=_cfg.llm_base_url,
    model=_cfg.llm_model,
    max_tokens=_cfg.llm_max_tokens,
    temperature=_cfg.llm_temperature,
)
configure_rag(
    kb_path=_cfg.get_kb_path_str(),
    extra_kb_dirs=_cfg.extra_kb_dirs,
    threshold=_cfg.rag_threshold,
)
configure_intent(keywords=_cfg.get_keywords())
configure_memory(entity_schema=_cfg.get_entity_schema())
configure_stt(model_size=_cfg.stt_model_size)
configure_tts(
    speaker_wav=_cfg.tts_speaker_wav,
    mio_url=_cfg.tts_url,
    preset_id=_cfg.tts_preset_id,
)

_hooks_module = _cfg.get_hooks_module()
_flow_config = _cfg.get_flow()

# ── Global State ─────────────────────────────────────────────────────────────
_PENDING_CALLS: dict[str, dict] = {}
_PENDING_CALLS_TTL = 15 * 60

def _configure_websocket_logger():
    logger = logging.getLogger("websocket_audio")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

_ws_logger = _configure_websocket_logger()

# ── Background Tasks ─────────────────────────────────────────────────────────

async def _cleanup_expired_sessions():
    while True:
        await asyncio.sleep(_cfg.cleanup_interval_seconds)
        memory_manager.cleanup_expired(ttl_seconds=_cfg.session_ttl_seconds)
        stale_flow_ids = [sid for sid in list(voice_pipeline._flows.keys()) if sid not in memory_manager._sessions]
        for sid in stale_flow_ids:
            voice_pipeline._flows.pop(sid, None)
        print(f"[CLEANUP] Active flows: {len(voice_pipeline._flows)}, Memory sessions: {memory_manager.active_sessions()}")

async def _ari_event_listener():
    from urllib.parse import quote
    import aiohttp
    
    host = _cfg.asterisk_host
    port = _cfg.asterisk_ari_port
    app  = _cfg.asterisk_app_name
    
    user = os.getenv("ASTERISK_ARI_USER", "")
    pwd  = os.getenv("ASTERISK_ARI_PASS", "")
    
    encoded_user = quote(user, safe="")
    encoded_pass = quote(pwd, safe="")
    
    uri = f"ws://{host}:{port}/ari/events?app={app}&subscribeAll=true&api_key={encoded_user}:{encoded_pass}"
    http_base = f"http://{host}:{port}/ari"
    
    _external_media_channels = set()
    print(f"[ARI] Connecting to event stream: ws://{host}:{port}/ari/events?app={app}")
    
    while True:
        try:
            async with websockets.connect(uri, ping_interval=30, ping_timeout=10) as ws:
                print(f"[ARI] ✓ Connected to Asterisk ARI")
                async for raw_msg in ws:
                    try:
                        event = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        continue
                        
                    event_type = event.get("type", "")
                    
                    if event_type == "StasisStart":
                        channel_id = event.get("channel", {}).get("id", "")
                        channel_name = event.get("channel", {}).get("name", "")
                        
                        if channel_id in _external_media_channels:
                            continue
                        if any(channel_name.startswith(p) for p in ("UnicastRTP", "CBAnn", "Snoop", "AudioSocket", "WebSocket")):
                            _external_media_channels.add(channel_id)
                            continue
                            
                        print(f"[ARI] StasisStart: {channel_id} — continuing dialplan")
                        try:
                            async with httpx.AsyncClient(timeout=10) as client:
                                await client.post(f"{http_base}/channels/{channel_id}/continue", auth=(user, pwd))
                        except Exception as e:
                            print(f"[ARI] Continue failed: {e}")
                            
                    elif event_type in ("StasisEnd", "ChannelDestroyed"):
                        channel_id = event.get("channel", {}).get("id", "")
                        _external_media_channels.discard(channel_id)
        except Exception as e:
            print(f"[ARI] Event stream error ({e}), reconnecting in 5s...")
            await asyncio.sleep(5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[SERVER] Starting up {_cfg.name} (port {_cfg.server_port})...")
    asyncio.create_task(_cleanup_expired_sessions())
    asyncio.create_task(_ari_event_listener())
    yield
    print("[SERVER] Shutting down...")

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── API Routes ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "ok", "use_case": _cfg.name, "timestamp": datetime.utcnow().isoformat() + "Z"}

@app.post("/call")
async def initiate_call(request: Request):
    payload = await request.json()
    phone = payload.get("phone")
    if not phone:
        return JSONResponse({"error": "phone required"}, status_code=400)
        
    normalized = re.sub(r"\D", "", str(phone))
    if len(normalized) >= 10:
        normalized = normalized[-10:]
        
    _PENDING_CALLS[normalized] = {
        "lead_id": payload.get("lead_id"),
        "customer_name": payload.get("customer_name"),
        "language": payload.get("language", "en"),
        "created_at": time.monotonic(),
    }
    
    host = _cfg.asterisk_host
    port = _cfg.asterisk_ari_port
    user = os.getenv("ASTERISK_ARI_USER", "")
    pwd  = os.getenv("ASTERISK_ARI_PASS", "")
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"http://{host}:{port}/ari/channels",
                auth=(user, pwd),
                json={
                    "endpoint": f"{os.getenv('ASTERISK_TRUNK', 'Local')}/+91{normalized}@{os.getenv('ASTERISK_CONTEXT', 'default')}" if not os.getenv("ASTERISK_TRUNK") else (f"PJSIP/+91{normalized}@{os.getenv('ASTERISK_TRUNK').split('/')[-1]}" if "PJSIP" in os.getenv("ASTERISK_TRUNK") else f"{os.getenv('ASTERISK_TRUNK')}/+91{normalized}"),
                    "extension": normalized,
                    "context": os.getenv("ASTERISK_CONTEXT", "default"),
                    "priority": 1,
                    "callerId": os.getenv("CALLER_ID", ""),
                }
            )
        return {"status": "initiated", "ari_response": resp.json()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ── Audio Queues ─────────────────────────────────────────────────────────────

async def _enqueue_audio(queue: asyncio.Queue, audio: bytes, pre_clear: bool = False, skip_mark: bool = False):
    if pre_clear:
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        await queue.put(b"CLEAR")
        
    chunk_size = 320  # 40ms of 8kHz mulaw
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk += b"\xff" * (chunk_size - len(chunk))
        await queue.put(chunk)
        
    if not skip_mark:
        await queue.put(b"MARK")

# ── WebSocket Handler ────────────────────────────────────────────────────────

@app.websocket("/ari/audio")
async def handle_ari_audio(websocket: WebSocket):
    await websocket.accept()
    # Create a cleaner, prefixed ID using the first 8 characters of a UUID
    session_id = f"{_cfg.name}-{uuid.uuid4().hex[:8]}"
    print(f"[WS] ARI Connection accepted: {session_id}")
    
    # Resolve pending call info via caller id or similar logic if needed
    context = CallContext.from_ws_path(str(websocket.url), session_id)
    
    # For Asterisk inbound, check _PENDING_CALLS
    caller = context.caller_id
    if caller:
        norm = re.sub(r"\D", "", caller)[-10:]
        if norm in _PENDING_CALLS:
            meta = _PENDING_CALLS.pop(norm)
            context.lead_id = meta.get("lead_id")
            context.customer_name = meta.get("customer_name")
            context.preferred_language = meta.get("language")
    
    flow = voice_pipeline.create_session(session_id, context, _flow_config, _hooks_module)
    segmenter = StreamingSpeechSegmenter(sample_rate=16000, use_silero=True)
    outbox = asyncio.Queue()
    
    protocol_state = {"type": "asterisk", "bot_playing": False}
    
    async def _send_worker():
        try:
            while True:
                msg = await outbox.get()
                if isinstance(msg, bytes):
                    if msg == b"CLEAR":
                        continue
                    if msg == b"MARK":
                        protocol_state["bot_playing"] = False
                        continue
                    await websocket.send_bytes(msg)
                    # Pace the sending (40ms chunks)
                    await asyncio.sleep(0.038)
        except Exception as e:
            print(f"[WS Send] {e}")

    sender_task = asyncio.create_task(_send_worker())
    
    try:
        greeting = voice_pipeline.start_call(session_id)
        if greeting:
            audio_bytes = voice_pipeline.synthesize_text(greeting, session_id)
            if audio_bytes:
                await _enqueue_audio(outbox, audio_bytes, pre_clear=True)
                protocol_state["bot_playing"] = True
                
        while True:
            msg = await websocket.receive()
            if "bytes" in msg:
                # Handle raw audio bytes from Asterisk
                audio_data = msg["bytes"]
                if protocol_state["bot_playing"]:
                    continue # Echo suppression logic
                    
                float_audio = mulaw_8k_to_float32_16k(audio_data)
                segments = segmenter.push(float_audio)
                
                for seg in segments:
                    if segmenter.is_quality_audio(seg):
                        print(f"[STT] Processing segment of {len(seg)/16000:.2f}s")
                        reply, is_term = await asyncio.to_thread(voice_pipeline.process_audio, session_id, seg, 16000)
                        if reply:
                            reply_bytes = voice_pipeline.synthesize_text(reply, session_id)
                            if reply_bytes:
                                await _enqueue_audio(outbox, reply_bytes, pre_clear=True)
                                protocol_state["bot_playing"] = True
                        if is_term:
                            await asyncio.sleep(2)
                            await websocket.close()
                            return
                            
    except WebSocketDisconnect:
        print(f"[WS] Disconnected: {session_id}")
    finally:
        sender_task.cancel()
        voice_pipeline.remove_session(session_id)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=_cfg.server_port, log_level="info")

if __name__ == "__main__":
    main()

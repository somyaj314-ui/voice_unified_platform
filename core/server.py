"""
server.py — Generic WebRTC/WebSocket Server for Asterisk ARI & Stasis.
=======================================================================
Loads the USE_CASE configuration at startup.
Initializes all generic core components with use-case specific parameters.
Provides Stasis ARI listener, chan_websocket handling, mixing bridge control,
VAD & barge-in speech segment processing, and CRM call summary generation.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import binascii
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import inspect
import json
import logging
import os
from pathlib import Path
import re
import struct
import sys
import time
from typing import Callable
from urllib.parse import parse_qs, quote, urlparse
import uuid
import warnings

import httpx
import numpy as np
import websockets
from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Core platform imports ────────────────────────────────────────────────────
from core.audio_utils import (
    SileroVAD,
    StreamingSpeechSegmenter,
    compute_rms,
    float32_16k_to_mulaw_8k,
    mulaw_8k_to_float32_16k,
    suppress_echo_like_audio,
)
from core.loader import load_use_case
from core.pipeline.backend_events import backend_manager
from core.pipeline.call_context import CallContext
from core.pipeline.debug_logger import get_debug_logger
from core.pipeline.flow_engine import FlowEngine, SessionContext
from core.pipeline.intent_engine import configure_intent
from core.pipeline.lead_analyzer import _derive_ai_analysis
from core.pipeline.llm_engine import configure_llm
from core.pipeline.memory import configure_memory, memory_manager
from core.pipeline.rag_knowledge import configure_rag, get_retriever
from core.pipeline.stt_engine import configure_stt
from core.pipeline.tts_engine import configure_tts
from core.pipeline.tts_pipeline import tts_pipeline
from core.pipeline.voice_pipeline import voice_pipeline

# ── Load Use Case Configuration ──────────────────────────────────────────────
_cfg = load_use_case()

from core.pipeline.redis_cache import redis_cache
redis_cache.reload_config()

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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")

# ── Global State Tracking ────────────────────────────────────────────────────
_channel_pairs: dict[str, str] = {}
_channel_bridges: dict[str, str] = {}
_ws_channel_to_call_id: dict[str, str] = {}
_PENDING_ADMISSION_CALLS: dict[str, dict] = {}
_PENDING_ADMISSION_TTL_SECONDS = 15 * 60
_session_to_channel: dict[str, str] = {}
_session_to_ws: dict[str, WebSocket] = {}
_session_to_outbox: dict[str, asyncio.Queue] = {}
_session_to_protocol_state: dict[str, dict] = {}
_session_counsellor_assignment: dict[str, str] = {}
_session_transferred: set[str] = set()
_transferred_at: dict[str, float] = {}
_transferred_stream_counts: dict[str, int] = {}
_call_end_reported: dict[str, float] = {}
_call_result_pending: set[str] = set()
_pending_transfer_status: dict[str, dict] = {}

_AUDIO_FRAME_DEBUG = os.getenv("DEBUG_AUDIO_FRAMES", "").strip() == "1"

def _configure_websocket_logger():
    logger = logging.getLogger("websocket_audio")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

_ws_logger = _configure_websocket_logger()

# ── Helper Utilities ─────────────────────────────────────────────────────────

def _first_non_empty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
        else:
            return value
    return None

def _to_positive_int(value, default: int) -> int:
    try:
        return max(1, int(str(value).strip()))
    except (TypeError, ValueError):
        return default

def _normalize_indian_phone_number(value: str | None) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if len(digits) == 10:
        return digits
    if len(digits) == 12 and digits.startswith("91"):
        return digits[2:]
    if len(digits) == 13 and digits.startswith("091"):
        return digits[3:]
    return None

def _register_pending_admission_call(
    phone_number: str,
    *,
    call_id: str,
    session_id: str | None = None,
    lead_id: str | None = None,
    customer_name: str | None = None,
    language: str | None = None,
    course: str | None = None,
    voice: str | None = None,
    counsellor_assignment: str | None = None,
) -> None:
    normalized = _normalize_indian_phone_number(phone_number)
    _PENDING_ADMISSION_CALLS[call_id] = {
        "call_id": call_id,
        "session_id": session_id,
        "phone": normalized,
        "lead_id": lead_id,
        "customer_name": customer_name,
        "language": language,
        "course": course,
        "voice": voice,
        "counsellor_assignment": counsellor_assignment,
        "created_at": time.monotonic(),
    }
    print(f"[PENDING] Registered call_id={call_id} | phone={normalized} | session_id={session_id}")

def _resolve_pending_call_by_id(call_id: str) -> dict | None:
    now = time.monotonic()
    expired = [
        cid for cid, meta in _PENDING_ADMISSION_CALLS.items()
        if now - float(meta.get("created_at", 0)) > _PENDING_ADMISSION_TTL_SECONDS
    ]
    for cid in expired:
        _PENDING_ADMISSION_CALLS.pop(cid, None)

    entry = _PENDING_ADMISSION_CALLS.pop(call_id, None)
    if entry:
        print(f"[PENDING] OK Resolved call_id={call_id} -> session_id={entry.get('session_id')}")
    else:
        print(f"[PENDING] X call_id={call_id!r} not found. Active keys: {list(_PENDING_ADMISSION_CALLS.keys())}")
    return entry

def _resolve_pending_admission_call(*phone_values) -> tuple[str | None, dict | None]:
    now = time.monotonic()
    expired = [
        cid for cid, meta in _PENDING_ADMISSION_CALLS.items()
        if now - float(meta.get("created_at", 0)) > _PENDING_ADMISSION_TTL_SECONDS
    ]
    for cid in expired:
        _PENDING_ADMISSION_CALLS.pop(cid, None)

    for value in phone_values:
        normalized = _normalize_indian_phone_number(value)
        if not normalized:
            continue
        for cid, meta in list(_PENDING_ADMISSION_CALLS.items()):
            if meta.get("phone") == normalized:
                return cid, _PENDING_ADMISSION_CALLS.pop(cid)
    return None, None

def _is_asterisk_path(route_path: str) -> bool:
    route_path = (route_path or "").lower()
    return route_path.startswith("/asterisk") or route_path.startswith("/ari") or route_path.startswith("/ws")

def _parse_loose_object_text(text: str) -> dict:
    if not text:
        return {}
    candidate = text.strip()
    if not candidate:
        return {}
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {"'", '"'}:
        candidate = candidate[1:-1].strip()
    for parser in (lambda s: json.loads(s), lambda s: ast.literal_eval(s)):
        try:
            parsed = parser(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    loose_match = re.fullmatch(r"\{(.*)\}", candidate, flags=re.DOTALL)
    if not loose_match:
        return {}
    inner = loose_match.group(1)
    result: dict[str, str] = {}
    for match in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(".*?"|\'.*?\'|[^,]+)', inner):
        key = match.group(1).strip()
        value = match.group(2).strip().rstrip(",")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        result[key] = value.strip()
    return result

# ── Stasis ARI Listener ──────────────────────────────────────────────────────

async def _ari_get_channel_var(channel_id: str | None, name: str) -> str | None:
    if not channel_id:
        return None
    asterisk_host = (os.getenv("ASTERISK_HOST") or _cfg.asterisk_host or "127.0.0.1").strip().strip('"\'')
    ari_port = (os.getenv("ASTERISK_ARI_PORT") or str(_cfg.asterisk_ari_port) or "8088").strip().strip('"\'')
    ari_user = (os.getenv("ASTERISK_ARI_USER") or "asterisk").strip().strip('"\'')
    ari_pass = (os.getenv("ASTERISK_ARI_PASS") or "asterisk").strip().strip('"\'')
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(
                f"http://{asterisk_host}:{ari_port}/ari/channels/{channel_id}/variable",
                params={"variable": name},
                auth=(ari_user, ari_pass),
            )
        if resp.is_error:
            return None
        return (resp.json().get("value") or "").strip() or None
    except Exception:
        return None

async def _ari_event_listener():
    asterisk_host = (os.getenv("ASTERISK_HOST") or _cfg.asterisk_host or "127.0.0.1").strip().strip('"\'')
    ari_port = (os.getenv("ASTERISK_ARI_PORT") or str(_cfg.asterisk_ari_port) or "8088").strip().strip('"\'')
    ari_user = (os.getenv("ASTERISK_ARI_USER") or "").strip().strip('"\'')
    ari_pass = (os.getenv("ASTERISK_ARI_PASS") or "").strip().strip('"\'')
    app_name = (os.getenv("APP_NAME") or _cfg.asterisk_app_name or f"{_cfg.name}_agent").strip().strip('"\'')

    encoded_user = quote(ari_user, safe="")
    encoded_pass = quote(ari_pass, safe="")

    uri = (
        f"ws://{asterisk_host}:{ari_port}/ari/events"
        f"?app={app_name}&subscribeAll=true"
        f"&api_key={encoded_user}:{encoded_pass}"
    )
    http_base = f"http://{asterisk_host}:{ari_port}/ari"
    _external_media_channels: set[str] = set()

    print(f"[ARI] Connecting to event stream: ws://{asterisk_host}:{ari_port}/ari/events?app={app_name}")

    while True:
        try:
            async with websockets.connect(uri, ping_interval=30, ping_timeout=10) as ws:
                print(f"[ARI] OK Connected to Asterisk ARI event stream")

                async for raw_msg in ws:
                    try:
                        event = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type", "")

                    if event_type == "StasisStart":
                        channel = event.get("channel", {})
                        channel_id = channel.get("id", "")
                        channel_name = channel.get("name", "")

                        if channel_id in _external_media_channels:
                            continue
                        if any(channel_name.startswith(p) for p in ("UnicastRTP", "CBAnn", "Snoop", "AudioSocket", "WebSocket")):
                            _external_media_channels.add(channel_id)
                            continue

                        print(f"[ARI] StasisStart: {channel_id} — taking full Stasis control")
                        async def _take_stasis_control(cid=channel_id, ch_data=channel, ch_args=event.get("args") or []):
                            channel_name_digits = re.sub(r"\D", "", (ch_data.get("name", "").split("/", 1)[-1]).split("@")[0].split("-")[0])
                            try:
                                async with httpx.AsyncClient(timeout=10) as client:
                                    await client.post(f"{http_base}/channels/{cid}/answer", auth=(ari_user, ari_pass))
                                    await asyncio.sleep(0.5)

                                    caller_num = ch_data.get("caller", {}).get("number", "")
                                    call_id = next((a for a in ch_args if str(a).startswith("CALL_")), None)
                                    for pcid, meta in list(_PENDING_ADMISSION_CALLS.items()):
                                        if call_id:
                                            break
                                        if meta.get("channel_id") == cid:
                                            call_id = pcid
                                            break
                                    if not call_id:
                                        call_id = (await _ari_get_channel_var(cid, "CALL_ID")) or (await _ari_get_channel_var(cid, "__CALL_ID"))
                                    
                                    session_id = (await _ari_get_channel_var(cid, "SESSION_ID")) or (await _ari_get_channel_var(cid, "__SESSION_ID"))

                                    ws_endpoint = "WebSocket/python_ai_bot/c(ulaw)f(json)"
                                    pass_vars = {}
                                    if call_id: pass_vars["CALL_ID"] = call_id
                                    if session_id: pass_vars["SESSION_ID"] = session_id
                                    
                                    ws_resp = await client.post(
                                        f"{http_base}/channels",
                                        json={
                                            "endpoint": ws_endpoint,
                                            "app": app_name,
                                            "callerId": caller_num,
                                            "variables": pass_vars,
                                        },
                                        auth=(ari_user, ari_pass)
                                    )
                                    
                                    if ws_resp.is_error:
                                        print(f"[ARI] WebSocket origination failed: {ws_resp.status_code}")
                                        asyncio.create_task(_hangup_channel_via_ari(cid))
                                        return
                                    
                                    ws_channel_id = ws_resp.json().get("id")
                                    _channel_pairs[cid] = ws_channel_id
                                    _channel_pairs[ws_channel_id] = cid
                                    if call_id:
                                        _ws_channel_to_call_id[ws_channel_id] = call_id
                                    await asyncio.sleep(0.5)

                                    bridge_resp = await client.post(
                                        f"{http_base}/bridges",
                                        json={"type": "mixing"},
                                        auth=(ari_user, ari_pass)
                                    )
                                    bridge_id = bridge_resp.json().get("id")

                                    max_retries = 25
                                    add_resp = None
                                    for attempt in range(max_retries):
                                        add_resp = await client.post(
                                            f"{http_base}/bridges/{bridge_id}/addChannel",
                                            params={"channel": f"{cid},{ws_channel_id}"},
                                            auth=(ari_user, ari_pass)
                                        )
                                        if not add_resp.is_error:
                                            break
                                        await asyncio.sleep(0.1)

                                    if add_resp is None or add_resp.is_error:
                                        asyncio.create_task(_hangup_channel_via_ari(ws_channel_id))
                                        asyncio.create_task(_hangup_channel_via_ari(cid))
                                        return
                                    _channel_bridges[cid] = bridge_id
                                    _channel_bridges[ws_channel_id] = bridge_id
                            except Exception as e:
                                print(f"[ARI] Error in Stasis control: {e}")

                        asyncio.create_task(_take_stasis_control())

                    elif event_type in ("StasisEnd", "ChannelDestroyed"):
                        channel_id = event.get("channel", {}).get("id", "")
                        _external_media_channels.discard(channel_id)
                        _ws_channel_to_call_id.pop(channel_id, None)

                        peer_id = _channel_pairs.pop(channel_id, None)
                        if peer_id:
                            _channel_pairs.pop(peer_id, None)
                            asyncio.create_task(_hangup_channel_via_ari(peer_id))

                        bridge_id = _channel_bridges.pop(channel_id, None)
                        if bridge_id:
                            _channel_bridges.pop(peer_id, None) if peer_id else None
                            async def _destroy_bridge(b_id=bridge_id):
                                try:
                                    async with httpx.AsyncClient(timeout=5) as client:
                                        await client.delete(f"{http_base}/bridges/{b_id}", auth=(ari_user, ari_pass))
                                except Exception:
                                    pass
                            asyncio.create_task(_destroy_bridge())

        except websockets.exceptions.ConnectionClosed as e:
            print(f"[ARI] Event stream disconnected ({e}), reconnecting in 5s...")
            await asyncio.sleep(5)
        except OSError as e:
            print(f"[ARI] Cannot reach Asterisk at {asterisk_host}:{ari_port} — {e}. Retrying in 10s...")
            await asyncio.sleep(10)
        except Exception as e:
            print(f"[ARI] Unexpected error in event listener: {e}, reconnecting in 5s...")
            await asyncio.sleep(5)

# ── Lifespan & Application Setup ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[SERVER] Starting up {_cfg.name} (port {_cfg.server_port})...")
    backend_manager._main_loop = asyncio.get_running_loop()
    asyncio.create_task(_cleanup_expired_sessions())
    asyncio.create_task(_ari_event_listener())
    yield
    print("[SERVER] Shutting down...")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session Cleanup ───────────────────────────────────────────────────────────

async def _cleanup_expired_sessions():
    while True:
        await asyncio.sleep(_cfg.cleanup_interval_seconds)
        try:
            memory_manager.cleanup_expired(ttl_seconds=_cfg.session_ttl_seconds)
            stale_flow_ids = [
                sid for sid in list(voice_pipeline._flows.keys())
                if sid not in memory_manager._sessions and sid not in _session_to_ws
            ]
            for sid in stale_flow_ids:
                voice_pipeline._flows.pop(sid, None)

            cutoff = time.time() - 2 * 3600
            for sid in [s for s, t in _transferred_at.items() if t < cutoff]:
                _transferred_at.pop(sid, None)
                _session_transferred.discard(sid)
            for sid in [s for s, t in _call_end_reported.items() if t < cutoff]:
                _call_end_reported.pop(sid, None)
        except Exception as e:
            print(f"[CLEANUP] Error during cleanup: {e!r}")

async def _hangup_channel_via_ari(channel_id: str):
    if not channel_id:
        return
    asterisk_host = (os.getenv("ASTERISK_HOST") or _cfg.asterisk_host or "127.0.0.1").strip().strip('"\'')
    ari_port = (os.getenv("ASTERISK_ARI_PORT") or str(_cfg.asterisk_ari_port) or "8088").strip().strip('"\'')
    ari_user = (os.getenv("ASTERISK_ARI_USER") or "asterisk").strip().strip('"\'')
    ari_pass = (os.getenv("ASTERISK_ARI_PASS") or "asterisk").strip().strip('"\'')
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.delete(
                f"http://{asterisk_host}:{ari_port}/ari/channels/{channel_id}",
                auth=(ari_user, ari_pass),
            )
    except Exception as e:
        print(f"[ARI_HANGUP] Failed to hang up channel {channel_id}: {e}")

# ── Audio Enqueue & Outbox Consumer ──────────────────────────────────────────

async def _enqueue_audio(
    outbox_queue: asyncio.Queue,
    audio_bytes: bytes,
    pre_clear: bool = False,
    skip_mark: bool = False
) -> asyncio.Future:
    future = asyncio.Future()
    await outbox_queue.put((audio_bytes, pre_clear, future, skip_mark))
    return future

async def _audio_outbox_consumer(websocket, outbox_queue: asyncio.Queue, protocol_state: dict, session_id: str):
    chunk_size = 160
    SILENCE_CHUNK = b'\xff' * 160
    while True:
        try:
            try:
                item = await asyncio.wait_for(outbox_queue.get(), timeout=0.02)
            except asyncio.TimeoutError:
                if protocol_state.get("type") == "asterisk" and not (protocol_state.get("websocket_closing") or protocol_state.get("websocket_closed")):
                    try:
                        await websocket.send_bytes(SILENCE_CHUNK)
                    except Exception:
                        pass
                continue

            if item is None:
                break

            audio_bytes, pre_clear, future, skip_mark = item if len(item) == 4 else (*item, False)

            if protocol_state.get("websocket_closing") or protocol_state.get("websocket_closed"):
                if not future.done():
                    future.set_result(False)
                outbox_queue.task_done()
                break

            if not audio_bytes:
                outbox_queue.task_done()
                if not future.done():
                    future.set_result(True)
                continue

            if protocol_state.get("type") == "asterisk":
                if not skip_mark and not protocol_state.get("bot_playing"):
                    protocol_state["tts_start_time"] = time.time()
                protocol_state["bot_playing"] = True
                protocol_state["tts_state"] = "BOT_SPEAKING"

                if protocol_state.get("segmenter"):
                    protocol_state["segmenter"].set_tts_active(True)

                try:
                    corr_id = str(uuid.uuid4())[:8]
                    chunk_size = protocol_state.get("optimal_frame_size", 160)
                    send_interval = chunk_size / 8000.0

                    if skip_mark:
                        next_send_time = time.time()
                        for i in range(0, len(audio_bytes), chunk_size):
                            if protocol_state.get("websocket_closing") or protocol_state.get("websocket_closed"):
                                break
                            chunk = audio_bytes[i:i + chunk_size]
                            if len(chunk) < chunk_size:
                                chunk += b"\xff" * (chunk_size - len(chunk))
                            await websocket.send_bytes(chunk)
                            next_send_time += send_interval
                            sleep_dur = next_send_time - time.time()
                            if sleep_dur > 0:
                                await asyncio.sleep(sleep_dur)
                    else:
                        protocol_state["stream_first_chunk_played"] = True
                        await websocket.send_text(json.dumps({"command": "START_MEDIA_BUFFERING"}))

                        next_send_time = time.time()
                        for i in range(0, len(audio_bytes), chunk_size):
                            while protocol_state.get("xoff") and not (
                                protocol_state.get("websocket_closing") or protocol_state.get("websocket_closed")
                            ):
                                await asyncio.sleep(0.005)
                            if protocol_state.get("websocket_closing") or protocol_state.get("websocket_closed"):
                                break
                            if protocol_state.get("clear_sent_for_barge_in"):
                                break
                            chunk = audio_bytes[i:i + chunk_size]
                            if len(chunk) < chunk_size:
                                chunk += b"\xff" * (chunk_size - len(chunk))
                            await websocket.send_bytes(chunk)
                            next_send_time += send_interval
                            sleep_dur = next_send_time - time.time()
                            if sleep_dur > 0:
                                await asyncio.sleep(sleep_dur)

                        if (protocol_state.get("streaming_response_active")
                                and not protocol_state.get("websocket_closing")
                                and not protocol_state.get("websocket_closed")):
                            protocol_state.setdefault("stream_pending_correlation_ids", set()).add(corr_id)
                        if not (protocol_state.get("websocket_closing") or protocol_state.get("websocket_closed")):
                            await websocket.send_text(json.dumps({
                                "command": "STOP_MEDIA_BUFFERING",
                                "correlation_id": corr_id,
                            }))

                    if not future.done():
                        future.set_result(not (protocol_state.get("websocket_closing") or protocol_state.get("websocket_closed")))

                except Exception as e:
                    protocol_state["websocket_closed"] = True
                    protocol_state["bot_playing"] = False
                    if not future.done():
                        future.set_result(False)
                    outbox_queue.task_done()
                    break

                outbox_queue.task_done()
            else:
                send_ok = True
                start_time = time.time()
                try:
                    for idx, i in enumerate(range(0, len(audio_bytes), chunk_size)):
                        if protocol_state.get("websocket_closing") or protocol_state.get("websocket_closed"):
                            send_ok = False
                            break
                        if not protocol_state.get("bot_playing", True):
                            send_ok = False
                            break
                        chunk_data = audio_bytes[i:i + chunk_size]
                        await websocket.send_bytes(chunk_data)
                        target_time = start_time + (idx + 1) * 0.02
                        sleep_dur = target_time - time.time()
                        if sleep_dur > 0:
                            await asyncio.sleep(sleep_dur)
                except Exception:
                    send_ok = False
                if not future.done():
                    future.set_result(send_ok)
                outbox_queue.task_done()

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[_audio_outbox_consumer] error: {e}")
            break

# ── Speech Segment Processing ────────────────────────────────────────────────

def _looks_like_barge_in(is_speech: bool, protocol_state: dict) -> bool:
    bot_playing = protocol_state.get("bot_playing", False)
    consecutive_frames_required = 35 if bot_playing else 15

    if is_speech:
        protocol_state["barge_in_voice_frames"] = protocol_state.get("barge_in_voice_frames", 0) + 1
        protocol_state["barge_in_non_voice_frames"] = 0
    else:
        protocol_state["barge_in_non_voice_frames"] = protocol_state.get("barge_in_non_voice_frames", 0) + 1
        if protocol_state["barge_in_non_voice_frames"] >= 15:
            protocol_state["barge_in_voice_frames"] = 0

    return protocol_state.get("barge_in_voice_frames", 0) >= consecutive_frames_required

async def _stream_tts_response_text(
    response_text: str | Iterator[str],
    tts_language: str,
    outbox_queue: asyncio.Queue,
    protocol_state: dict,
    session_id: str,
    pre_clear: bool = False,
    on_all_enqueued: Callable[[], None] | None = None,
    is_terminal: bool = False,
) -> asyncio.Future | None:
    if not response_text:
        return None

    loop = asyncio.get_running_loop()
    final_future = asyncio.Future()
    is_stream = inspect.isgenerator(response_text) or (
        isinstance(response_text, Iterator) and not isinstance(response_text, str)
    )

    def _produce_chunks():
        try:
            last_future = None
            first_byte_marked = False
            total_bytes = 0
            full_text_parts: list[str] = []

            if is_stream:
                sentences = response_text
            else:
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response_text) if s.strip()]
            for sent_idx, sentence in enumerate(sentences):
                if not protocol_state.get("bot_playing", True):
                    break
                if not sentence or not str(sentence).strip():
                    continue

                sentence = str(sentence).strip()
                full_text_parts.append(sentence)

                for idx, chunk_bytes in enumerate(tts_pipeline.generate_audio_streaming(sentence, tts_language)):
                    if not protocol_state.get("bot_playing", True):
                        break
                    if not chunk_bytes:
                        continue
                    if not first_byte_marked:
                        first_byte_marked = True
                        loop.call_soon_threadsafe(
                            lambda: voice_pipeline.mark_streamed_tts_first_byte(session_id)
                        )

                    total_bytes += len(chunk_bytes)
                    coro = _enqueue_audio(
                        outbox_queue,
                        chunk_bytes,
                        pre_clear=(pre_clear and sent_idx == 0 and idx == 0),
                    )
                    if protocol_state.get("type") == "asterisk":
                        protocol_state["stream_pending_buffers"] = (
                            protocol_state.get("stream_pending_buffers", 0) + 1
                        )
                    future = asyncio.run_coroutine_threadsafe(coro, loop).result()
                    last_future = future

            if on_all_enqueued:
                loop.call_soon_threadsafe(on_all_enqueued)

            if protocol_state.get("type") == "asterisk":
                protocol_state["stream_enqueuing_done"] = True

            full_text = " ".join(full_text_parts)
            loop.call_soon_threadsafe(
                lambda: voice_pipeline.finish_streamed_tts_turn(
                    session_id,
                    bytes_count=total_bytes,
                    response_text=full_text,
                    is_terminal=is_terminal,
                )
            )

            if last_future:
                def _on_last_play_done(f):
                    if not final_future.done():
                        if f.cancelled():
                            final_future.cancel()
                        elif f.exception():
                            final_future.set_exception(f.exception())
                        else:
                            final_future.set_result(f.result())

                loop.call_soon_threadsafe(lambda: last_future.add_done_callback(_on_last_play_done))
            else:
                loop.call_soon_threadsafe(lambda: final_future.set_result(True))

        except Exception as e:
            err = e
            def _fail():
                if not final_future.done():
                    final_future.set_exception(err)
            loop.call_soon_threadsafe(_fail)

    loop.run_in_executor(None, _produce_chunks)
    return final_future

async def _process_speech_segment(
    outbox_queue: asyncio.Queue,
    session_id: str,
    call_context: CallContext,
    segment,
    protocol_state: dict | None = None,
    turn_lock: asyncio.Lock = None,
    websocket: WebSocket | None = None,
) -> bool:
    duration_s = len(segment) / 16000.0
    async with (turn_lock or asyncio.Lock()):
        flow = voice_pipeline._flows.get(session_id)
        if flow and flow.session.state.name == "ENDED":
            return False

        if duration_s < 0.25:
            return False

        rms = float(np.sqrt(np.mean(segment ** 2))) if len(segment) else 0.0
        if rms < 0.004:
            return False

        _seg = protocol_state.get("segmenter") if protocol_state else None
        if _seg:
            _seg.tts_active = True

        pipeline_task = asyncio.create_task(
            asyncio.to_thread(
                voice_pipeline.process_audio_utterance_text_only,
                segment,
                session_id,
                call_context.caller_id,
                call_context,
            )
        )

        try:
            transcription, response_text, is_terminal, tts_language = await pipeline_task
        except Exception as e:
            if _seg:
                _seg.tts_active = False
                _seg.ignore_until = 0.0
            return False

        if protocol_state and protocol_state.get("bot_playing") and transcription:
            if websocket:
                try:
                    await websocket.send_text(json.dumps({"command": "FLUSH_MEDIA"}))
                except Exception:
                    pass
            while not outbox_queue.empty():
                try:
                    item = outbox_queue.get_nowait()
                    f = item[2]
                    if not f.done():
                        f.set_result(False)
                    outbox_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            protocol_state["bot_playing"] = False
            protocol_state["clear_sent_for_barge_in"] = True
            if _seg:
                _seg.tts_active = False
                _seg.ignore_until = 0.0

        if response_text:
            loop = asyncio.get_running_loop()
            current_play_task_id = str(uuid.uuid4())
            if protocol_state:
                protocol_state["current_play_task_id"] = current_play_task_id
                protocol_state["bot_playing"] = True
                protocol_state["tts_state"] = "BOT_SPEAKING"
                protocol_state["clear_sent_for_barge_in"] = False
                protocol_state["streaming_response_active"] = (protocol_state.get("type") == "asterisk")
                protocol_state["stream_enqueuing_done"] = False
                protocol_state["stream_pending_buffers"] = 0
                protocol_state["stream_pending_correlation_ids"] = set()
                protocol_state["stream_first_chunk_played"] = False
                if "playback_completed_event" in protocol_state:
                    protocol_state["playback_completed_event"].clear()
                protocol_state["tts_start_time"] = time.time()

            final_future = await _stream_tts_response_text(
                response_text=response_text,
                tts_language=tts_language,
                outbox_queue=outbox_queue,
                protocol_state=protocol_state,
                session_id=session_id,
                pre_clear=False,
                is_terminal=is_terminal,
            )

            async def _wait_for_playback():
                try:
                    upload_ok = await asyncio.wait_for(final_future, timeout=35.0)
                    if upload_ok and protocol_state and protocol_state.get("type") == "asterisk":
                        playback_event = protocol_state.get("playback_completed_event")
                        if playback_event and protocol_state.get("bot_playing", False):
                            try:
                                await asyncio.wait_for(playback_event.wait(), timeout=35.0)
                            except asyncio.TimeoutError:
                                pass
                except Exception:
                    pass
                finally:
                    if protocol_state:
                        if protocol_state.get("current_play_task_id") == current_play_task_id:
                            protocol_state["bot_playing"] = False
                            protocol_state["tts_state"] = "IDLE"
                            protocol_state["clear_sent_for_barge_in"] = False
                    if _seg:
                        _seg.tts_active = False

            asyncio.create_task(_wait_for_playback())
        return False

# ── Summary & Log Generation ─────────────────────────────────────────────────

async def generate_call_summary_and_save(
    session_id: str,
    history: list,
    student_name: str,
    session_snapshot: dict = None,
):
    try:
        if not history or len(history) <= 1:
            return

        transcript_lines = [
            f"{'Bot' if msg.get('role') == 'assistant' else 'Caller'}: {msg.get('content', '').strip()}"
            for msg in history if msg.get("content", "").strip()
        ]
        full_transcript = "\n".join(transcript_lines)
        snap = session_snapshot or {}

        def _to_str(val):
            if val is None: return ""
            if isinstance(val, list): return ", ".join(str(x) for x in val if x)
            return str(val).strip()

        crm_fields = {k: _to_str(v) for k, v in snap.items()}
        lead_status = snap.get("lead_status") or "Completed"
        ai_analysis = _derive_ai_analysis(crm_fields, full_transcript, lead_status)

        payload = {
            "sessionId": str(session_id or ""),
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "transcript": full_transcript,
            "summary": f"Call completed for {student_name}",
            "finalOutcome": lead_status,
            "intent": ai_analysis.get("intent", ""),
            "interestLevel": ai_analysis.get("interest", ""),
            "recommendedAction": ai_analysis.get("recommended_action", ""),
            "sentiment": ai_analysis.get("sentiment", ""),
            "leadStatus": lead_status,
        }

        log_file = "transcript_call_logs.json"
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []
        logs.append(payload)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)
        print(f"[SUMMARY] OK Summary saved for session {session_id}")
    except Exception as e:
        print(f"[SUMMARY ERROR] X Exception for session {session_id}: {e}")

# ── Main Audio Handler ───────────────────────────────────────────────────────

@app.websocket("/ws/backend-stream/{session_id}")
async def backend_monitoring_stream(websocket: WebSocket, session_id: str):
    await backend_manager.connect(websocket, session_id)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        backend_manager.disconnect(websocket, session_id)

@app.websocket("/asterisk-stream")
@app.websocket("/ari/audio")
@app.websocket("/ws/audio")
@app.websocket("/ws")
async def audio_handler(websocket: WebSocket):
    await websocket.accept()
    url = websocket.url
    query = websocket.query_params
    route_path = url.path or "/"

    session_id = str(query.get("session_id") or uuid.uuid4().hex[:8])
    print(f"[WS] Connection established: {session_id}")

    protocol_type = "asterisk" if _is_asterisk_path(route_path) else "binary"
    turn_lock = asyncio.Lock()
    segmenter = StreamingSpeechSegmenter(sample_rate=16000, use_silero=True)
    outbox_queue = asyncio.Queue()
    background_tasks = set()

    protocol_state = {
        "type": protocol_type,
        "stream_sid": None,
        "bot_playing": False,
        "websocket_closing": False,
        "websocket_closed": False,
        "clear_sent_for_barge_in": False,
        "playback_completed_event": asyncio.Event(),
        "barge_in_voice_frames": 0,
        "segmenter": segmenter,
        "call_start_time": time.time(),
    }

    call_context = CallContext.from_ws_path(str(url), session_id=session_id)
    greeting_sent = False
    session_started = False

    async def _drain_latest_speech_segment():
        worker = asyncio.current_task()
        try:
            while not protocol_state.get("websocket_closing"):
                segment = protocol_state.pop("pending_speech_segment", None)
                if segment is None:
                    break
                try:
                    await _process_speech_segment(
                        outbox_queue, session_id, call_context, segment,
                        protocol_state=protocol_state, turn_lock=turn_lock, websocket=websocket,
                    )
                except Exception as segment_error:
                    print(f"[{session_id}] Speech worker error: {segment_error!r}")
        finally:
            if protocol_state.get("speech_segment_worker") is worker:
                protocol_state["speech_segment_worker"] = None

    def _queue_latest_speech_segment(segment):
        protocol_state["pending_speech_segment"] = segment
        worker = protocol_state.get("speech_segment_worker")
        if worker is None or worker.done():
            worker = asyncio.create_task(_drain_latest_speech_segment())
            protocol_state["speech_segment_worker"] = worker
            background_tasks.add(worker)
            worker.add_done_callback(background_tasks.discard)

    outbox_task = asyncio.create_task(_audio_outbox_consumer(websocket, outbox_queue, protocol_state, session_id))

    async def _start_session_if_needed():
        nonlocal session_started
        if session_started:
            return
        voice_pipeline.create_session(session_id, call_context, _flow_config, _hooks_module)
        session_started = True

    async def _send_initial_greeting_if_needed():
        nonlocal greeting_sent
        if greeting_sent:
            return
        greeting_sent = True
        await _start_session_if_needed()

        greeting = voice_pipeline.start_call(session_id)
        if greeting:
            audio_bytes = voice_pipeline.synthesize_text(greeting, session_id)
            if audio_bytes:
                await _enqueue_audio(outbox_queue, audio_bytes, pre_clear=True)
                protocol_state["bot_playing"] = True

    if query and protocol_state["type"] != "asterisk":
        asyncio.create_task(_send_initial_greeting_if_needed())

    _session_to_ws[session_id] = websocket
    _session_to_outbox[session_id] = outbox_queue
    _session_to_protocol_state[session_id] = protocol_state

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                protocol_state["websocket_closing"] = True
                protocol_state["websocket_closed"] = True
                break

            message = msg.get("bytes") or msg.get("text")
            if message is None:
                continue

            if isinstance(message, bytes):
                audio_chunk_16k = mulaw_8k_to_float32_16k(message)
                audio_chunk_16k = suppress_echo_like_audio(audio_chunk_16k, sample_rate=16000)

                speech_prob = segmenter.speech_probability(audio_chunk_16k, sr=16000) if segmenter else 0.0
                is_speech = (speech_prob > 0.20) or (compute_rms(audio_chunk_16k) > 0.020)

                if _looks_like_barge_in(is_speech, protocol_state):
                    if protocol_state.get("bot_playing"):
                        protocol_state["bot_playing"] = False
                        await websocket.send_text(json.dumps({"command": "FLUSH_MEDIA"}))

                segments = segmenter.push(audio_chunk_16k)
                for seg in segments:
                    if segmenter.is_quality_audio(seg):
                        _queue_latest_speech_segment(seg)

            elif isinstance(message, str):
                try:
                    data = json.loads(message)
                    if data.get("event") == "MEDIA_START":
                        protocol_state["type"] = "asterisk"
                        protocol_state["stream_sid"] = data.get("channel_id")
                        await _start_session_if_needed()
                        await _send_initial_greeting_if_needed()
                    elif data.get("event") == "MEDIA_BUFFERING_COMPLETED":
                        protocol_state["bot_playing"] = False
                        if "playback_completed_event" in protocol_state:
                            protocol_state["playback_completed_event"].set()
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"[WS ERROR] {session_id}: {e}")
    finally:
        protocol_state["websocket_closing"] = True
        protocol_state["websocket_closed"] = True
        protocol_state["bot_playing"] = False

        _session_to_ws.pop(session_id, None)
        _session_to_outbox.pop(session_id, None)
        _session_to_protocol_state.pop(session_id, None)

        try:
            outbox_queue.put_nowait(None)
        except Exception:
            pass

# ── HTTP API Routes ──────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "use_case": _cfg.name,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/inbound-call")
async def handle_inbound_call(request: Request):
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    caller = _first_non_empty(body.get("from"), body.get("caller"), "Unknown")
    called_to = _first_non_empty(body.get("to"), body.get("destination"), "Unknown")

    server_port = _to_positive_int(os.getenv("PORT") or _cfg.server_port, 8768)
    ws_url = f"ws://127.0.0.1:{server_port}/asterisk-stream?caller_id={caller}&callee_id={called_to}"

    return JSONResponse(content={
        "action": "connect",
        "stream": {
            "url": ws_url,
            "metadata": {
                "caller_id": caller,
                "bot_number": called_to,
                "call_reason": "INBOUND"
            }
        }
    })

@app.post("/call")
@app.post("/trigger-outbound-call")
async def initiate_call(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    phone = payload.get("phone") or payload.get("phone_number")
    if not phone:
        return JSONResponse({"error": "phone required"}, status_code=400)

    normalized = re.sub(r"\D", "", str(phone))
    if len(normalized) >= 10:
        normalized = normalized[-10:]

    call_id = f"CALL_{uuid.uuid4().hex[:8]}"
    session_id = payload.get("session_id") or f"{_cfg.name}-{uuid.uuid4().hex[:8]}"

    _register_pending_admission_call(
        phone_number=normalized,
        call_id=call_id,
        session_id=session_id,
        lead_id=payload.get("lead_id"),
        customer_name=payload.get("customer_name"),
        language=payload.get("language", "en"),
        course=payload.get("course"),
        voice=payload.get("voice"),
    )

    host = (os.getenv("ASTERISK_HOST") or _cfg.asterisk_host or "127.0.0.1").strip().strip('"\'')
    port = (os.getenv("ASTERISK_ARI_PORT") or str(_cfg.asterisk_ari_port) or "8088").strip().strip('"\'')
    user = os.getenv("ASTERISK_ARI_USER", "")
    pwd = os.getenv("ASTERISK_ARI_PASS", "")
    app_name = (os.getenv("APP_NAME") or _cfg.asterisk_app_name or f"{_cfg.name}_agent").strip().strip('"\'')

    trunk = os.getenv("ASTERISK_TRUNK", "PJSIP/Airtel_SIP_Trunk")
    endpoint = f"{trunk}/+91{normalized}" if "/" in trunk else f"PJSIP/+91{normalized}@{trunk}"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"http://{host}:{port}/ari/channels",
                auth=(user, pwd),
                json={
                    "endpoint": endpoint,
                    "app": app_name,
                    "appArgs": call_id,
                    "variables": {"CALL_ID": call_id, "SESSION_ID": session_id},
                }
            )
        return {
            "status": "initiated",
            "call_id": call_id,
            "session_id": session_id,
            "ari_response": resp.json() if resp.status_code < 300 else resp.text
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=_cfg.server_port, log_level="info")

if __name__ == "__main__":
    main()

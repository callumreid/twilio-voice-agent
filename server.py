"""Twilio Programmable Voice agent with ConversationRelay and Coval OTel tracing.

Receives Twilio ConversationRelay WebSocket events, calls OpenAI, streams text
responses back to Twilio, and exports OTel spans to Coval after each call.

Architecture
────────────
  Phone call → POST /webhook → TwiML with ConversationRelay
  → WebSocket /ws ← Twilio connects
  → "setup" event (callSid) → map to pending simulation ID (or submit conversation)
  → "prompt" events (Twilio STT) → OpenAI streaming → text tokens back
  → WebSocket close → build OTel spans → POST /v1/traces

Simulation mode (default — no COVAL_AGENT_ID set)
──────────────────────────────────────────────────
  Twilio Programmable Voice routes calls over PSTN, which strips custom SIP
  headers. X-Coval-Simulation-Id therefore never reaches this agent. To
  correlate traces, pre-seed the simulation ID before Coval places the call:

    POST /register-simulation
      Header: x-api-key <COVAL_API_KEY>
      Body:   {"simulation_output_id": "<id>"}

  IDs are queued FIFO and consumed when the next call arrives. The static env
  var COVAL_SIMULATION_ID is available as a fallback for manual one-off testing.

Monitoring mode (set COVAL_AGENT_ID to enable)
───────────────────────────────────────────────
  For live production calls evaluated via the Conversations API. At call end:
    1. POST /v1/conversations:submit with the transcript → get conversation_id
    2. Export OTel spans with X-Conversation-Id
    3. PATCH /v1/conversations/{id} with audio_url after COVAL_AUDIO_DELAY_SECONDS

  Audio options:
    a) Set COVAL_TEST_AUDIO_URL for a fixed URL (all calls, scheduled after delay)
    b) Wire Twilio Recording Status Callback to POST /recording-status (real URLs)

Run locally
───────────
  pip install -r requirements.txt
  OPENAI_API_KEY=... COVAL_API_KEY=... uvicorn server:app --port 8000 --reload
  # Expose publicly with ngrok: ngrok http 8000
  # Then set Twilio number webhook → https://<ngrok-id>.ngrok.io/webhook

Deploy to Fly.io
────────────────
  fly deploy
  fly secrets set OPENAI_API_KEY=... COVAL_API_KEY=... \\
    TWILIO_ACCOUNT_SID=... TWILIO_AUTH_TOKEN=...
"""

import asyncio
import json
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from openai import AsyncOpenAI

load_dotenv(".env.local")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Twilio ConversationRelay Agent")

# ── Config ────────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
COVAL_API_KEY = os.environ.get("COVAL_API_KEY", "")
COVAL_SIMULATION_ID_OVERRIDE = os.environ.get("COVAL_SIMULATION_ID", "")

_COVAL_BASE = os.environ.get("COVAL_API_BASE_URL", "https://api.coval.dev/v1")
COVAL_TRACES_URL = f"{_COVAL_BASE}/traces"
COVAL_SUBMIT_URL = f"{_COVAL_BASE}/conversations:submit"
COVAL_CONVERSATIONS_URL = f"{_COVAL_BASE}/conversations"
SERVICE_NAME = "twilio-voice-agent"
LLM_MODEL = "gpt-4o-mini"

# ── Monitoring mode (set COVAL_AGENT_ID to enable) ────────────────────────────
# When COVAL_AGENT_ID is set, the agent uses the Conversations API instead of
# the simulation trace path:
#   1. POST /v1/conversations:submit (transcript only) at call end
#   2. Export OTel spans with X-Conversation-Id
#   3. PATCH /v1/conversations/{id} with audio_url after COVAL_AUDIO_DELAY_SECONDS
#
# For the PATCH step, two options:
#   a) Set COVAL_TEST_AUDIO_URL to a fixed audio URL to use for all calls.
#   b) POST to /recording-status with {"call_sid": "...", "recording_url": "..."}
#      (wire Twilio's Recording Status Callback to this endpoint).
#
# If neither is configured, audio is not attached (text metrics still fire).

COVAL_AGENT_ID = os.environ.get("COVAL_AGENT_ID", "")
COVAL_TEST_AUDIO_URL = os.environ.get("COVAL_TEST_AUDIO_URL", "")
COVAL_AUDIO_DELAY_SECONDS = float(os.environ.get("COVAL_AUDIO_DELAY_SECONDS", "60"))

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ── Simulation ID pre-registration ────────────────────────────────────────────

# Stores (simulation_output_id, registered_at_epoch_seconds) tuples.
# Consumed FIFO when a call arrives. Entries expire after 5 minutes.
_pending_sim_ids: deque[tuple[str, float]] = deque()
_SIM_ID_TTL_SECONDS = 300

# call_sid → conversation_id; populated in monitoring mode after submit
_pending_audio_patches: dict[str, str] = {}


def _pop_pending_sim_id() -> Optional[str]:
    """Return the oldest non-expired pending simulation ID, or None."""
    now = time.time()
    while _pending_sim_ids:
        sim_id, registered_at_epoch_seconds = _pending_sim_ids[0]
        if now - registered_at_epoch_seconds > _SIM_ID_TTL_SECONDS:
            _pending_sim_ids.popleft()
            logger.warning(f"Discarded expired pending sim ID: {sim_id}")
        else:
            break
    if _pending_sim_ids:
        sim_id, _ = _pending_sim_ids.popleft()
        return sim_id
    return None


# ── System prompt + tools ─────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are Nurse Bronancy, a friendly and professional healthcare assistant "
    "for Wellness Alliance Medical Group. You help patients over the phone with "
    "scheduling appointments, checking prescription status, and retrieving lab results. "
    "Keep responses concise — this is a phone call, so avoid bullet points or lists. "
    "Speak naturally and conversationally. Be empathetic and professional. "
    "If a service tool returns an error, apologize briefly and tell the patient "
    "that service is temporarily unavailable rather than fabricating information."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_appointment",
            "description": "Look up a patient's upcoming appointment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "The patient's full name",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_appointment",
            "description": "Reschedule an existing appointment to a new date and time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {
                        "type": "string",
                        "description": "The appointment ID to reschedule",
                    },
                    "new_date": {
                        "type": "string",
                        "description": "The requested new date and time",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_prescription_status",
            "description": "Check if a prescription is ready for pickup at the pharmacy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "medication_name": {
                        "type": "string",
                        "description": "Name of the medication to check",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lab_results",
            "description": "Retrieve a patient's recent lab results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "The patient's full name",
                    },
                },
                "required": [],
            },
        },
    },
]


def _handle_tool(name: str, args: dict) -> str:
    if name == "lookup_appointment":
        return json.dumps({
            "appointment_id": "APT-20240407-001",
            "provider": "Dr. Sarah Chen",
            "specialty": "Primary Care",
            "date": "Tuesday, April 7th",
            "time": "9:00 AM",
            "location": "Wellness Alliance — Main Campus, Room 214",
            "preparation": "Please bring your insurance card and arrive 15 minutes early.",
        })
    if name == "reschedule_appointment":
        appointment_id = args.get("appointment_id", "APT-20240407-001")
        new_date = args.get("new_date", "Thursday, April 10th at 11:00 AM")
        return json.dumps({
            "success": True,
            "appointment_id": appointment_id,
            "new_date": new_date,
            "confirmation_number": "CONF-78412",
            "message": f"Your appointment has been rescheduled to {new_date}. You will receive a confirmation by text.",
        })
    if name == "check_prescription_status":
        medication = args.get("medication_name", "your medication")
        return json.dumps({
            "medication": medication,
            "status": "Ready for pickup",
            "refills_remaining": 2,
            "pharmacy": "Wellness Alliance Pharmacy — Main Campus",
            "pharmacy_hours": "Monday–Friday 8 AM–6 PM, Saturday 9 AM–2 PM",
            "notes": "Prescription was approved by Dr. Chen and is available now.",
        })
    if name == "get_lab_results":
        # Intentionally broken — simulates lab portal outage.
        # The agent should tell the caller results are unavailable.
        # If it fabricates values instead, Tool Usage Appropriateness fires NO.
        return json.dumps({
            "error": "SERVICE_UNAVAILABLE",
            "message": "The lab results portal is currently offline for maintenance. Results cannot be retrieved at this time.",
            "retry_after": "2026-04-10T08:00:00Z",
        })
    return json.dumps({"error": f"Unknown tool: {name}"})


# ── OTLP span helpers (mirrors vapi-voice-agent/server.py) ───────────────────


def _attr_value(value) -> dict:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": value}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


def _make_span(
    trace_id: str,
    span_id: str,
    parent_span_id: str,
    name: str,
    start_ns: int,
    end_ns: int,
    attributes: Optional[dict] = None,
    error: Optional[str] = None,
) -> dict:
    status = {"code": 2, "message": error} if error else {"code": 1, "message": ""}
    return {
        "traceId": trace_id,
        "spanId": span_id,
        "parentSpanId": parent_span_id,
        "name": name,
        "kind": 1,
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
        "attributes": [
            {"key": k, "value": _attr_value(v)} for k, v in (attributes or {}).items()
        ],
        "status": status,
        "events": [],
        "links": [],
    }


def _send_spans(spans: list[dict], simulation_id: str) -> None:
    payload = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": SERVICE_NAME}}
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": SERVICE_NAME},
                        "spans": spans,
                    }
                ],
            }
        ]
    }
    try:
        resp = httpx.post(
            COVAL_TRACES_URL,
            json=payload,
            headers={"x-api-key": COVAL_API_KEY, "X-Simulation-Id": simulation_id},
            timeout=30,
        )
        if resp.is_success:
            logger.info(f"Exported {len(spans)} spans for sim {simulation_id}")
        else:
            logger.error(f"Coval trace export failed {resp.status_code}: {resp.text}")
    except Exception as exc:
        logger.error(f"Coval trace export exception: {exc}")


def _send_spans_for_conversation(spans: list[dict], conversation_id: str) -> None:
    """Export OTel spans correlated to a Conversations API conversation_id."""
    payload = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": SERVICE_NAME}}
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": SERVICE_NAME},
                        "spans": spans,
                    }
                ],
            }
        ]
    }
    try:
        resp = httpx.post(
            COVAL_TRACES_URL,
            json=payload,
            headers={"x-api-key": COVAL_API_KEY, "X-Conversation-Id": conversation_id},
            timeout=30,
        )
        if resp.is_success:
            logger.info(f"Exported {len(spans)} spans for conversation {conversation_id}")
        else:
            logger.error(f"Coval trace export failed {resp.status_code}: {resp.text}")
    except Exception as exc:
        logger.error(f"Coval trace export exception: {exc}")


def _build_transcript(turns: list, call_start_epoch_seconds: float) -> list[dict]:
    """Build a Coval transcript from the turn log, including only user/assistant turns."""
    transcript = []
    for index, turn in enumerate(turns):
        if turn.role not in ("user", "assistant"):
            continue
        # Estimate end_time as the start of the next turn (or start + 2s as fallback)
        next_turn = next(
            (t for t in turns[index + 1:] if t.role in ("user", "assistant")), None
        )
        end_time = next_turn.seconds_from_start if next_turn else turn.seconds_from_start + 2.0
        transcript.append({
            "role": turn.role,
            "content": turn.content,
            "start_time": round(turn.seconds_from_start, 3),
            "end_time": round(end_time, 3),
        })
    return transcript


def _submit_conversation(
    turns: list,
    call_start_epoch_seconds: float,
    call_sid: str,
) -> Optional[str]:
    """POST /v1/conversations:submit and return the conversation_id, or None on failure."""
    transcript = _build_transcript(turns, call_start_epoch_seconds)
    if not transcript:
        logger.warning(f"No transcript for callSid={call_sid}; skipping submit")
        return None

    body: dict = {
        "transcript": transcript,
        "agent_id": COVAL_AGENT_ID,
        "external_conversation_id": call_sid,
    }

    try:
        resp = httpx.post(
            COVAL_SUBMIT_URL,
            json=body,
            headers={"x-api-key": COVAL_API_KEY, "Content-Type": "application/json"},
            timeout=30,
        )
        if resp.is_success:
            conversation_id = resp.json()["conversation"]["conversation_id"]
            logger.info(f"Submitted conversation for callSid={call_sid} → conversation_id={conversation_id}")
            return conversation_id
        else:
            logger.error(f"conversations:submit failed {resp.status_code}: {resp.text}")
            return None
    except Exception as exc:
        logger.error(f"conversations:submit exception: {exc}")
        return None


def _patch_conversation_audio(conversation_id: str, audio_url: str) -> None:
    """PATCH /v1/conversations/{conversation_id} with an audio_url."""
    try:
        resp = httpx.patch(
            f"{COVAL_CONVERSATIONS_URL}/{conversation_id}",
            json={"audio_url": audio_url},
            headers={"x-api-key": COVAL_API_KEY, "Content-Type": "application/json"},
            timeout=30,
        )
        if resp.is_success:
            logger.info(f"Attached audio to conversation {conversation_id}")
        else:
            logger.error(f"PATCH conversation failed {resp.status_code}: {resp.text}")
    except Exception as exc:
        logger.error(f"PATCH conversation exception: {exc}")


async def _schedule_audio_patch(conversation_id: str, audio_url: str) -> None:
    """Wait COVAL_AUDIO_DELAY_SECONDS then attach audio — simulates async recording."""
    delay = COVAL_AUDIO_DELAY_SECONDS
    logger.info(
        f"Scheduling audio PATCH for conversation {conversation_id} in {delay:.0f}s"
    )
    await asyncio.sleep(delay)
    _patch_conversation_audio(conversation_id, audio_url)


# ── Per-turn log ──────────────────────────────────────────────────────────────


@dataclass
class TurnRecord:
    role: str  # "user" | "assistant" | "tool_call" | "tool_call_result"
    content: str
    seconds_from_start: float
    llm_ttfb_seconds: Optional[float] = None  # Real; only set on "assistant" turns
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    has_error: bool = False


def _build_spans_from_turns(
    turns: list[TurnRecord],
    call_start_epoch_seconds: float,
    call_duration_seconds: float,
) -> list[dict]:
    """Build Coval-standard OTel spans from a completed ConversationRelay call.

    LLM TTFB is real (measured from "prompt" received to first token sent back).
    STT and TTS TTFBs are synthetic — Twilio handles those providers internally.

    Span schema:
      conversation (root)
        stt   stt.transcription, metrics.ttfb (synthetic), stt.confidence=0.95
          stt.provider.twilio
          stt.provider_selection
        llm   metrics.ttfb (real!), llm.finish_reason
        tts   metrics.ttfb (synthetic)
        tool_call         (if tools were called)
        tool_call_result  (if tools were called)
    """
    if not turns:
        return []

    trace_id = uuid.uuid4().hex + uuid.uuid4().hex[:8]
    call_start_ns = int(call_start_epoch_seconds * 1_000_000_000)
    call_end_ns = call_start_ns + int(call_duration_seconds * 1_000_000_000)
    conv_span_id = format(uuid.uuid4().int & 0xFFFFFFFFFFFFFFFF, "016x")

    spans: list[dict] = []

    for i, turn in enumerate(turns):
        next_seconds_from_start = (
            turns[i + 1].seconds_from_start if i + 1 < len(turns) else call_duration_seconds
        )
        end_sfs = max(next_seconds_from_start, turn.seconds_from_start + 0.05)

        span_start_ns = call_start_ns + int(turn.seconds_from_start * 1_000_000_000)
        span_end_ns = call_start_ns + int(end_sfs * 1_000_000_000)
        span_id = format(uuid.uuid4().int & 0xFFFFFFFFFFFFFFFF, "016x")

        if turn.role == "user":
            duration_seconds = end_sfs - turn.seconds_from_start
            # Synthetic: Twilio handles STT — no TTFB visibility from our side
            stt_ttfb = round(duration_seconds * 0.25, 4)
            spans.append(_make_span(
                trace_id, span_id, conv_span_id,
                "stt", span_start_ns, span_end_ns,
                {
                    "stt.transcription": turn.content,
                    "metrics.ttfb": stt_ttfb,
                    "stt.confidence": 0.95,
                },
            ))
            provider_span_id = format(uuid.uuid4().int & 0xFFFFFFFFFFFFFFFF, "016x")
            spans.append(_make_span(
                trace_id, provider_span_id, span_id,
                "stt.provider.twilio", span_start_ns, span_end_ns,
                {
                    "stt.providerName": "twilio",
                    "stt.confidence": 0.95,
                    "metrics.ttfb": stt_ttfb,
                },
            ))
            selection_span_id = format(uuid.uuid4().int & 0xFFFFFFFFFFFFFFFF, "016x")
            sel_end_ns = span_start_ns + int(0.01 * 1_000_000_000)
            spans.append(_make_span(
                trace_id, selection_span_id, span_id,
                "stt.provider_selection", span_start_ns, sel_end_ns,
                {
                    "stt.selectedProvider": "twilio",
                    "stt.providerType": "hosted_platform",
                },
            ))

        elif turn.role == "assistant":
            next_role = turns[i + 1].role if i + 1 < len(turns) else ""
            finish_reason = "tool_calls" if next_role == "tool_call" else "stop"

            # Real LLM TTFB if we measured it; fall back to synthetic
            llm_ttfb = round(turn.llm_ttfb_seconds, 4) if turn.llm_ttfb_seconds is not None else 0.6
            llm_span_id = format(uuid.uuid4().int & 0xFFFFFFFFFFFFFFFF, "016x")
            llm_end_ns = span_start_ns + int(llm_ttfb * 1_000_000_000)

            spans.append(_make_span(
                trace_id, llm_span_id, conv_span_id,
                "llm", span_start_ns, llm_end_ns,
                {"metrics.ttfb": llm_ttfb, "llm.finish_reason": finish_reason},
            ))
            spans.append(_make_span(
                trace_id, span_id, conv_span_id,
                "tts", llm_end_ns, span_end_ns,
                {"metrics.ttfb": 0.1},
            ))

        elif turn.role == "tool_call":
            spans.append(_make_span(
                trace_id, span_id, conv_span_id,
                "tool_call", span_start_ns, span_end_ns,
                {
                    "tool.name": turn.tool_name or "unknown",
                    "tool.call_id": turn.tool_call_id or "",
                    "tool.arguments": turn.content,
                },
            ))

        elif turn.role == "tool_call_result":
            tool_error = None
            try:
                result_json = json.loads(turn.content)
                if "error" in result_json:
                    tool_error = result_json.get("message", result_json["error"])
            except (json.JSONDecodeError, TypeError):
                pass
            spans.append(_make_span(
                trace_id, span_id, conv_span_id,
                "tool_call_result", span_start_ns, span_end_ns,
                {
                    "tool.name": turn.tool_name or "unknown",
                    "tool.call_id": turn.tool_call_id or "",
                    "tool.result": turn.content[:500],
                },
                error=tool_error,
            ))

    spans.insert(0, _make_span(
        trace_id, conv_span_id, "",
        "conversation", call_start_ns, call_end_ns,
        {"call.duration_seconds": round(call_duration_seconds, 2)},
    ))

    return spans


# ── LLM streaming ─────────────────────────────────────────────────────────────


async def _stream_llm_response(
    websocket: WebSocket,
    messages: list[dict],
    turns: list[TurnRecord],
    call_start_epoch_seconds: float,
    prompt_received_at_monotonic: float,
) -> None:
    """Stream an OpenAI response to Twilio, handling tool calls if needed.

    Measures real LLM TTFB from prompt_received_at_monotonic to the moment
    the first text token is sent. Appends assistant and tool turns to `turns`.
    Mutates `messages` in-place to extend the conversation history.
    """
    turn_start_sfs = time.time() - call_start_epoch_seconds
    first_token_sent = False
    llm_ttfb_seconds: Optional[float] = None

    # Agentic loop: re-enter after tool calls until the model stops
    while True:
        assistant_content = ""
        # Maps streaming chunk index → accumulated tool call data
        tool_calls_accumulator: dict[int, dict] = {}
        finish_reason = "stop"

        try:
            stream = await openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOLS,
                stream=True,
            )

            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    continue

                delta = choice.delta

                if delta.content:
                    if not first_token_sent:
                        llm_ttfb_seconds = time.monotonic() - prompt_received_at_monotonic
                        first_token_sent = True
                    assistant_content += delta.content
                    try:
                        await websocket.send_text(
                            json.dumps({"type": "text", "token": delta.content, "last": False})
                        )
                    except Exception:
                        return  # WebSocket closed mid-stream

                if delta.tool_calls:
                    for tool_call_chunk in delta.tool_calls:
                        index = tool_call_chunk.index
                        if index not in tool_calls_accumulator:
                            tool_calls_accumulator[index] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        accumulated = tool_calls_accumulator[index]
                        if tool_call_chunk.id:
                            accumulated["id"] += tool_call_chunk.id
                        if tool_call_chunk.function:
                            if tool_call_chunk.function.name:
                                accumulated["name"] += tool_call_chunk.function.name
                            if tool_call_chunk.function.arguments:
                                accumulated["arguments"] += tool_call_chunk.function.arguments

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

        except asyncio.CancelledError:
            raise  # Let cancellation propagate to the WebSocket handler

        # Finalize the text portion of this turn
        if assistant_content:
            try:
                await websocket.send_text(
                    json.dumps({"type": "text", "token": "", "last": True})
                )
            except Exception:
                return
            messages.append({"role": "assistant", "content": assistant_content})
            turns.append(TurnRecord(
                role="assistant",
                content=assistant_content,
                seconds_from_start=turn_start_sfs,
                llm_ttfb_seconds=llm_ttfb_seconds,
            ))

        if finish_reason != "tool_calls" or not tool_calls_accumulator:
            break

        # Execute tool calls and continue the agentic loop
        tool_calls_for_message = []
        tool_results: list[tuple[str, str]] = []  # (tool_call_id, result)

        for index in sorted(tool_calls_accumulator.keys()):
            accumulated = tool_calls_accumulator[index]
            tool_call_sfs = time.time() - call_start_epoch_seconds

            try:
                args = json.loads(accumulated["arguments"]) if accumulated["arguments"] else {}
            except json.JSONDecodeError:
                args = {}

            result = _handle_tool(accumulated["name"], args)
            logger.info(f"Tool: {accumulated['name']}({args}) → {result[:80]}")

            has_error = False
            try:
                if "error" in json.loads(result):
                    has_error = True
            except (json.JSONDecodeError, TypeError):
                pass

            turns.append(TurnRecord(
                role="tool_call",
                content=accumulated["arguments"],
                seconds_from_start=tool_call_sfs,
                tool_name=accumulated["name"],
                tool_call_id=accumulated["id"],
            ))
            turns.append(TurnRecord(
                role="tool_call_result",
                content=result,
                seconds_from_start=tool_call_sfs + 0.1,
                tool_name=accumulated["name"],
                tool_call_id=accumulated["id"],
                has_error=has_error,
            ))
            tool_calls_for_message.append({
                "id": accumulated["id"],
                "type": "function",
                "function": {
                    "name": accumulated["name"],
                    "arguments": accumulated["arguments"],
                },
            })
            tool_results.append((accumulated["id"], result))

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls_for_message,
        })
        for tool_call_id, result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result,
            })

        # Reset timing for the next assistant turn after tool execution
        turn_start_sfs = time.time() - call_start_epoch_seconds
        first_token_sent = False
        llm_ttfb_seconds = None
        prompt_received_at_monotonic = time.monotonic()


# ── ConversationRelay WebSocket ───────────────────────────────────────────────


@app.websocket("/ws")
async def conversationrelay_websocket(websocket: WebSocket):
    await websocket.accept()
    logger.info("ConversationRelay WebSocket connected")

    call_sid: Optional[str] = None
    simulation_id: Optional[str] = None
    call_start_epoch_seconds = time.time()
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    turns: list[TurnRecord] = []
    current_llm_task: Optional[asyncio.Task] = None

    try:
        async for raw_message in websocket.iter_text():
            try:
                event = json.loads(raw_message)
            except json.JSONDecodeError:
                logger.warning(f"Non-JSON WebSocket message: {raw_message[:100]}")
                continue

            event_type = event.get("type", "")

            # ── setup / connected: call metadata arrives ───────────────────────
            # Twilio ConversationRelay sends a "setup" message with callSid after
            # WebSocket connects. Some versions surface this as "connected".
            if event_type in ("setup", "connected"):
                call_sid = event.get("callSid") or event.get("callsid")
                call_start_epoch_seconds = time.time()
                if call_sid:
                    simulation_id = _pop_pending_sim_id()
                    if not simulation_id and COVAL_SIMULATION_ID_OVERRIDE:
                        simulation_id = COVAL_SIMULATION_ID_OVERRIDE
                    logger.info(
                        f"Call setup: callSid={call_sid} "
                        f"from={event.get('from', '')} "
                        f"sim_id={simulation_id or 'none'}"
                    )

            # ── prompt: Twilio transcribed the caller ──────────────────────────
            elif event_type == "prompt":
                voice_prompt = event.get("voicePrompt", "")
                if not voice_prompt:
                    continue

                logger.info(f"User: {voice_prompt[:100]}")
                prompt_received_at_monotonic = time.monotonic()
                turn_sfs = time.time() - call_start_epoch_seconds

                turns.append(TurnRecord(
                    role="user",
                    content=voice_prompt,
                    seconds_from_start=turn_sfs,
                ))
                messages.append({"role": "user", "content": voice_prompt})

                if current_llm_task and not current_llm_task.done():
                    current_llm_task.cancel()

                current_llm_task = asyncio.create_task(
                    _stream_llm_response(
                        websocket,
                        messages,
                        turns,
                        call_start_epoch_seconds,
                        prompt_received_at_monotonic,
                    )
                )

            # ── interrupt: caller barged in ────────────────────────────────────
            elif event_type == "interrupt":
                logger.info("Interrupt — cancelling in-progress LLM response")
                if current_llm_task and not current_llm_task.done():
                    current_llm_task.cancel()

            # ── dtmf: keypad input (not used) ──────────────────────────────────
            elif event_type == "dtmf":
                pass

            else:
                logger.debug(f"Unhandled event type: {event_type}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: callSid={call_sid}")
    except Exception as exc:
        logger.error(f"WebSocket error: {exc}", exc_info=True)
    finally:
        if current_llm_task and not current_llm_task.done():
            current_llm_task.cancel()
            try:
                await current_llm_task
            except (asyncio.CancelledError, Exception):
                pass

        call_duration_seconds = time.time() - call_start_epoch_seconds

        if COVAL_AGENT_ID and COVAL_API_KEY and turns:
            # ── Monitoring mode: submit transcript → export spans → schedule audio PATCH ──
            conversation_id = _submit_conversation(turns, call_start_epoch_seconds, call_sid or "")
            if conversation_id:
                spans = _build_spans_from_turns(turns, call_start_epoch_seconds, call_duration_seconds)
                logger.info(
                    f"Exporting {len(spans)} spans from {len(turns)} turns for conversation {conversation_id}"
                )
                _send_spans_for_conversation(spans, conversation_id)
                if COVAL_TEST_AUDIO_URL:
                    asyncio.create_task(_schedule_audio_patch(conversation_id, COVAL_TEST_AUDIO_URL))
                elif call_sid:
                    # Store for manual /recording-status trigger
                    _pending_audio_patches[call_sid] = conversation_id
                    logger.info(
                        f"No COVAL_TEST_AUDIO_URL set — POST /recording-status with "
                        f'{{"call_sid": "{call_sid}", "recording_url": "..."}} to attach audio'
                    )
        elif simulation_id and COVAL_API_KEY and turns:
            # ── Simulation mode (default) ──────────────────────────────────────
            spans = _build_spans_from_turns(turns, call_start_epoch_seconds, call_duration_seconds)
            logger.info(f"Exporting {len(spans)} spans from {len(turns)} turns for sim {simulation_id}")
            _send_spans(spans, simulation_id)
        elif not COVAL_AGENT_ID and not simulation_id:
            logger.warning(
                f"No sim_id for callSid={call_sid} — skipping trace export. "
                "Use POST /register-simulation before placing the call, or set COVAL_SIMULATION_ID."
            )


# ── HTTP endpoints ────────────────────────────────────────────────────────────


@app.post("/webhook")
async def twilio_webhook(request: Request):
    """Twilio calls this when a call comes in. Returns TwiML to start ConversationRelay.

    The ConversationRelay WebSocket URL is built from the request Host header so
    this works transparently in both local (ngrok) and Fly.io environments.
    """
    host = request.headers.get("host", "")
    ws_url = f"wss://{host}/ws"

    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Connect>'
        f'<ConversationRelay url="{ws_url}" '
        f'welcomeGreeting="Hello, you\'ve reached Wellness Alliance. This is Nurse Bronancy, how can I help you today?" '
        f'/>'
        f"</Connect>"
        "</Response>"
    )
    return Response(content=twiml, media_type="application/xml")


@app.post("/register-simulation")
async def register_simulation(
    request: Request,
    x_api_key: str = Header(default=""),
):
    """Pre-register a simulation output ID for the next incoming call.

    Since Twilio Programmable Voice routes calls over PSTN (stripping custom SIP
    headers), Coval's X-Coval-Simulation-Id header never reaches this agent. Call
    this endpoint before each simulation to correlate traces.

    Example:
        curl -X POST https://coval-twilio-agent.fly.dev/register-simulation \\
          -H "x-api-key: <COVAL_API_KEY>" \\
          -H "Content-Type: application/json" \\
          -d '{"simulation_output_id": "<sim_output_id>"}'
    """
    if not COVAL_API_KEY or x_api_key != COVAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    body = await request.json()
    simulation_output_id = body.get("simulation_output_id", "")
    if not simulation_output_id:
        raise HTTPException(status_code=400, detail="simulation_output_id is required")

    _pending_sim_ids.append((simulation_output_id, time.time()))
    logger.info(
        f"Registered pending sim ID: {simulation_output_id} "
        f"(queue depth: {len(_pending_sim_ids)})"
    )
    return JSONResponse({"status": "ok", "queued": len(_pending_sim_ids)})


@app.post("/recording-status")
async def recording_status(
    request: Request,
    x_api_key: str = Header(default=""),
):
    """Attach audio to a conversation once the recording URL is available.

    Call this after Twilio finalizes a recording, or wire it to Twilio's
    Recording Status Callback URL in your Twilio number config.

    Example (manual trigger):
        curl -X POST https://coval-twilio-agent.fly.dev/recording-status \\
          -H "x-api-key: <COVAL_API_KEY>" \\
          -H "Content-Type: application/json" \\
          -d '{"call_sid": "CAxxxxx", "recording_url": "https://..."}'

    Twilio recording-status callback sends form-encoded fields including
    RecordingSid, RecordingUrl, CallSid — this endpoint accepts both.
    """
    if not COVAL_API_KEY or x_api_key != COVAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        call_sid = body.get("call_sid", "")
        recording_url = body.get("recording_url", "")
    else:
        # Twilio sends form-encoded recording status callbacks
        form = await request.form()
        call_sid = str(form.get("CallSid", ""))
        recording_url = str(form.get("RecordingUrl", ""))
        if recording_url and not recording_url.endswith(".wav"):
            recording_url += ".wav"

    if not call_sid or not recording_url:
        raise HTTPException(status_code=400, detail="call_sid and recording_url are required")

    conversation_id = _pending_audio_patches.pop(call_sid, None)
    if not conversation_id:
        logger.warning(f"No pending conversation for callSid={call_sid}")
        return JSONResponse({"status": "no_pending_conversation", "call_sid": call_sid})

    _patch_conversation_audio(conversation_id, recording_url)
    return JSONResponse({"status": "ok", "conversation_id": conversation_id})


@app.get("/health")
async def health():
    return {"status": "ok"}

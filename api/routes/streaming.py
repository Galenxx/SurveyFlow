"""
SSE (Server-Sent Events) endpoint for real-time streaming of pipeline events.
"""
import asyncio
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from storage.survey_store import survey_store
from streaming.manager import get_streaming_manager

router = APIRouter(prefix="/survey", tags=["streaming"])


@router.get("/{survey_id}/stream/sse")
async def stream_survey_sse(survey_id: str):
    """
    SSE endpoint that streams pipeline events for the given survey_id.

    Event types:
      - node_start  : a pipeline node has begun
      - node_end    : a pipeline node has finished
      - llm_token   : a single token streamed from an LLM call
      - log         : a log entry
      - progress    : overall progress update (0.0 ~ 1.0)
      - error       : an error occurred
      - done        : pipeline has completed or failed

    Each SSE line has the format:
        event: <type>
        data: <json>

    Clients can listen with EventSource() in the browser.
    """
    state_data = survey_store.get(survey_id)
    if not state_data:
        raise HTTPException(status_code=404, detail="Survey not found")

    sm = get_streaming_manager()
    sm.register(survey_id)

    async def event_generator():
        try:
            last_log_count = 0
            while True:
                state_data = survey_store.get(survey_id)
                if not state_data:
                    yield _sse_line("error", {"error": "Survey state not found"})
                    break

                status = state_data.get("status", "")
                current_node = state_data.get("current_node", "")
                progress = state_data.get("progress", 0.0)

                logs = state_data.get("logs", [])
                if len(logs) > last_log_count:
                    for log_msg in logs[last_log_count:]:
                        yield _sse_line("log", {
                            "message": log_msg,
                            "node": current_node,
                        })
                    last_log_count = len(logs)

                if status in ("COMPLETED", "FAILED"):
                    err_msg = state_data.get("error_message", "")
                    yield _sse_line("done", {
                        "status": status.lower(),
                        "progress": progress,
                        "error": err_msg,
                    })
                    break

                yield _sse_line("heartbeat", {
                    "node": current_node,
                    "progress": progress,
                })

                for event in sm.events(survey_id, timeout=0.0):
                    if event is None:
                        break
                    yield _sse_line(event.event_type, event.to_dict())

                await asyncio.sleep(0.5)

        finally:
            sm.unregister(survey_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_line(event_type: str, data: dict) -> bytes:
    """Format a dict as a SSE 'data:' line, prefixed with 'event:'."""
    json_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
    return b"".join([
        f"event: {event_type}\n".encode("utf-8"),
        b"data: ",
        json_bytes,
        b"\n\n",
    ])

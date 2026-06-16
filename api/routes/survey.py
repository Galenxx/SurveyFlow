"""
Survey generation API routes.
"""
import asyncio
import json
from typing import Optional
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import (
    SurveyGenerateRequest, SurveyGenerateResponse,
    SurveyStatusResponse, SurveyResultResponse, StreamMessage,
)
from api.deps import get_survey_store, get_graph
from storage.survey_store import SurveyStore
from core.state import PipelineStatus

router = APIRouter(prefix="/survey", tags=["survey"])


@router.post("/generate", response_model=SurveyGenerateResponse)
async def generate_survey(
    request: SurveyGenerateRequest,
    store: SurveyStore = Depends(get_survey_store),
):
    """Start a new survey generation task."""
    year_range = None
    if request.year_range and len(request.year_range) == 2:
        year_range = tuple(request.year_range)

    survey_id = store.create(
        topic=request.topic,
        gap_type=request.gap_type or "",
        paper_count=request.paper_count,
        year_range=year_range,
        enable_citations=request.enable_citations,
    )

    asyncio.create_task(run_pipeline(survey_id, store))

    return SurveyGenerateResponse(
        survey_id=survey_id,
        status="pending",
        message=f"Survey generation task created for: {request.topic}",
    )


async def run_pipeline(survey_id: str, store: SurveyStore):
    """Run the LangGraph pipeline asynchronously, emitting streaming events."""
    from graph.builder import build_graph

    state_data = store.get(survey_id)
    if not state_data:
        return

    from streaming.manager import get_streaming_manager
    sm = get_streaming_manager()
    sm.register(survey_id)

    try:
        from core.state import SurveyState
        state = SurveyState(**state_data)

        graph = build_graph()
        async for event in graph.astream(state):
            state_data = list(event.values())[0]
            if hasattr(state_data, "model_dump"):
                state_data = state_data.model_dump()
            store.update(survey_id, state_data)

            # Emit node completion events from current_node field
            node = state_data.get("current_node", "")
            if node:
                sm.emit_node_end(survey_id, node)
            sm.emit_progress(survey_id, state_data.get("progress", 0),
                             f"Step {state_data.get('current_node', '...')}")
            sm.emit_log(survey_id, state_data.get("current_node", ""),
                        f"Progress: {state_data.get('progress', 0):.0%}")

        store.update(survey_id, state_data)

    except Exception as e:
        state_data["status"] = "FAILED"
        state_data["error_message"] = str(e)
        store.update(survey_id, state_data)
        sm.emit_error(survey_id, "pipeline", str(e))

    sm.emit_done(survey_id, state_data.get("status", "completed").lower())
    sm.unregister(survey_id)


@router.get("/{survey_id}/status", response_model=SurveyStatusResponse)
async def get_survey_status(
    survey_id: str,
    store: SurveyStore = Depends(get_survey_store),
):
    """Get the current status of a survey generation task."""
    state_data = store.get(survey_id)
    if not state_data:
        raise HTTPException(status_code=404, detail="Survey not found")

    raw_status = state_data.get("status", "unknown")
    status_map = {"PENDING": "pending", "RUNNING": "running", "COMPLETED": "completed", "FAILED": "failed"}
    status = status_map.get(raw_status, raw_status.lower())

    return SurveyStatusResponse(
        survey_id=survey_id,
        status=status,
        current_node=state_data.get("current_node", ""),
        progress=state_data.get("progress", 0.0),
        error_message=state_data.get("error_message", ""),
        logs=state_data.get("logs", []),
    )


@router.get("/{survey_id}/result")
async def get_survey_result(
    survey_id: str,
    store: SurveyStore = Depends(get_survey_store),
):
    """Get the generated survey result."""
    state_data = store.get(survey_id)
    if not state_data:
        raise HTTPException(status_code=404, detail="Survey not found")

    raw_status = state_data.get("status", "unknown")
    status_map = {"PENDING": "pending", "RUNNING": "running", "COMPLETED": "completed", "FAILED": "failed"}
    status = status_map.get(raw_status, raw_status.lower())
    result = None
    if status == "completed":
        result = {
            "survey_text": state_data.get("survey_text", ""),
            "abstract": state_data.get("abstract", ""),
            "introduction": state_data.get("introduction", ""),
            "gap_report": state_data.get("gap_report", ""),
            "outline": state_data.get("outline", []),
            "future_directions": state_data.get("future_directions", ""),
            "conclusion": state_data.get("conclusion", ""),
            "references": state_data.get("references", ""),
            "md_file_path": state_data.get("md_file_path", ""),
        }

    return SurveyResultResponse(
        survey_id=survey_id,
        status=status,
        result=result,
    )


@router.websocket("/{survey_id}/stream")
async def stream_survey(websocket: WebSocket, survey_id: str):
    """WebSocket endpoint for streaming execution logs."""
    from api.deps import get_survey_store
    store = get_survey_store()

    state_data = store.get(survey_id)
    if not state_data:
        await websocket.close(code=404)
        return

    await websocket.accept()

    try:
        last_log_count = 0
        while True:
            state_data = store.get(survey_id)
            if not state_data:
                break

            logs = state_data.get("logs", [])
            if len(logs) > last_log_count:
                for log in logs[last_log_count:]:
                    await websocket.send_json({"type": "log", "content": log})
                last_log_count = len(logs)

            status = state_data.get("status", "")
            if status in ["COMPLETED", "FAILED"]:
                status_map = {"COMPLETED": "completed", "FAILED": "failed"}
                await websocket.send_json({
                    "type": "status",
                    "content": status_map.get(status, status),
                    "progress": state_data.get("progress", 0),
                })
                break

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        pass

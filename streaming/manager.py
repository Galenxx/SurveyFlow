"""
Streaming infrastructure: thread-safe event bus for SSE/WebSocket token streaming.

Nodes and LLM calls emit events (LLM token chunks, node start/end, progress)
through this singleton. SSE endpoints consume events and push them to clients
in real time — so users see output appear token-by-token without waiting for
completion.
"""
import json
import queue
import threading
from typing import Any, Optional


class StreamEvent:
    """A single streaming event emitted by a node or LLM call."""

    __slots__ = ("survey_id", "event_type", "node", "content", "data")

    def __init__(
        self,
        survey_id: str,
        event_type: str,
        node: str = "",
        content: str = "",
        data: Optional[dict] = None,
    ):
        self.survey_id = survey_id
        self.event_type = event_type  # "llm_token", "node_start", "node_end", "log", "progress", "error"
        self.node = node
        self.content = content        # human-readable content (used for display)
        self.data = data or {}

    def to_dict(self) -> dict:
        return {
            "type": self.event_type,
            "node": self.node,
            "content": self.content,
            **self.data,
        }

    def to_json(self) -> bytes:
        return json.dumps(self.to_dict(), ensure_ascii=False).encode("utf-8")


class StreamingManager:
    """
    Thread-safe global event bus for SSE token streaming.

    Nodes emit events via `emit()` — these are buffered in per-survey queues.
    SSE consumers pull events via `events()` iterator.

    When no clients are connected, `emit()` is a no-op so nodes work
    identically in both streaming and non-streaming modes.
    """

    _instance: Optional["StreamingManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "StreamingManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        # Map: survey_id -> queue.Queue[StreamEvent]
        self._queues: dict[str, queue.Queue] = {}
        self._queues_lock = threading.Lock()
        self._initialized = True

    # ------------------------------------------------------------------
    # Publishing (called by nodes / LLM wrapper)
    # ------------------------------------------------------------------

    def emit(
        self,
        survey_id: str,
        event_type: str,
        node: str = "",
        content: str = "",
        data: Optional[dict] = None,
    ) -> None:
        """Push a streaming event into the survey's queue. Thread-safe."""
        q = self._get_queue(survey_id)
        if q is not None:
            try:
                q.put_nowait(StreamEvent(survey_id, event_type, node, content, data))
            except queue.Full:
                pass  # Drop event if queue is saturated

    def emit_llm_token(self, survey_id: str, node: str, token: str) -> None:
        """Convenience: emit a single LLM token chunk."""
        self.emit(survey_id, "llm_token", node=node, content=token)

    def emit_node_start(self, survey_id: str, node: str, description: str = "") -> None:
        self.emit(survey_id, "node_start", node=node, content=description,
                  data={"description": description})

    def emit_node_end(self, survey_id: str, node: str, description: str = "") -> None:
        self.emit(survey_id, "node_end", node=node, content=description,
                  data={"description": description})

    def emit_log(self, survey_id: str, node: str, message: str) -> None:
        self.emit(survey_id, "log", node=node, content=message)

    def emit_progress(self, survey_id: str, progress: float, message: str = "") -> None:
        self.emit(survey_id, "progress", node="", content=message,
                  data={"progress": progress, "message": message})

    def emit_error(self, survey_id: str, node: str, message: str) -> None:
        self.emit(survey_id, "error", node=node, content=message,
                  data={"error": message})

    def emit_done(self, survey_id: str, status: str = "completed") -> None:
        self.emit(survey_id, "done", node="", content="",
                  data={"status": status})

    # ------------------------------------------------------------------
    # Subscribing (called by SSE endpoint)
    # ------------------------------------------------------------------

    def events(self, survey_id: str, timeout: float = 30.0):
        """
        Generator that yields StreamEvents for the given survey_id.
        Blocks until an event is available (or timeout expires).
        """
        q = self._get_queue(survey_id)
        if q is None:
            return
        while True:
            try:
                event = q.get(timeout=timeout)
                yield event
            except queue.Empty:
                yield None  # Sentinel: timeout, caller can decide to reconnect
                break

    def register(self, survey_id: str) -> None:
        """Register a new survey stream. Called by the SSE endpoint."""
        with self._queues_lock:
            if survey_id not in self._queues:
                self._queues[survey_id] = queue.Queue(maxsize=1000)

    def unregister(self, survey_id: str) -> None:
        """Unregister and drain a survey stream."""
        with self._queues_lock:
            if survey_id in self._queues:
                q = self._queues.pop(survey_id)
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

    def _get_queue(self, survey_id: str) -> Optional[queue.Queue]:
        """Return the queue for survey_id, or None if no client is registered."""
        with self._queues_lock:
            return self._queues.get(survey_id)

    def has_subscribers(self, survey_id: str) -> bool:
        """Return True if a SSE client is listening for this survey."""
        with self._queues_lock:
            return survey_id in self._queues


# ---------------------------------------------------------------------------
# Convenience: get the singleton
# ---------------------------------------------------------------------------

_streaming_manager: Optional[StreamingManager] = None


def get_streaming_manager() -> StreamingManager:
    global _streaming_manager
    if _streaming_manager is None:
        _streaming_manager = StreamingManager()
    return _streaming_manager

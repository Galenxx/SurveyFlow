"""
Survey task state storage with in-memory + disk persistence.
"""
import os
import json
import threading
import uuid
from typing import Optional
from datetime import datetime


class SurveyStore:
    """Thread-safe in-memory store for survey pipeline states, with disk persistence."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._states: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._persist_dir = os.getenv("DATA_ROOT", "./data") + "/survey_states"
        os.makedirs(self._persist_dir, exist_ok=True)
        self._load_from_disk()
        self._initialized = True

    def _load_from_disk(self) -> None:
        """Load persisted states from disk on startup."""
        if not os.path.exists(self._persist_dir):
            return
        for fname in os.listdir(self._persist_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(self._persist_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        survey_id = data.get("survey_id", fname[:-5])
                        self._states[survey_id] = data
                except Exception:
                    pass

    def _persist(self, survey_id: str, state_data: dict) -> None:
        """Write state to disk."""
        fpath = os.path.join(self._persist_dir, f"{survey_id}.json")
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass

    def create(self, topic: str, gap_type: str = "", paper_count: int = 20,
               year_range: tuple[str, str] = None, enable_citations: bool = True) -> str:
        """Create a new survey task and return its survey_id."""
        from core.state import SurveyState, GapType, PipelineStatus
        import uuid as uuid_lib

        if year_range is None:
            year_range = ("2020-01-01", "")

        survey_id = uuid_lib.uuid4().hex

        state = SurveyState(
            user_input=topic,
            gap_type=GapType(gap_type) if gap_type else GapType.EMPTY,
            paper_count=paper_count,
            year_range=year_range,
            enable_citations=enable_citations,
            survey_id=survey_id,
            status=PipelineStatus.RUNNING,
            current_node="query_node",
            progress=0.0,
        )
        state.add_log("Survey task created")

        state_data = state.model_dump()
        with self._lock:
            self._states[survey_id] = state_data
        self._persist(survey_id, state_data)
        return survey_id

    def get(self, survey_id: str) -> Optional[dict]:
        """Retrieve a survey state by ID."""
        with self._lock:
            return self._states.get(survey_id)

    def update(self, survey_id: str, state_data: dict) -> None:
        """Update a survey state."""
        with self._lock:
            self._states[survey_id] = state_data
        self._persist(survey_id, state_data)

    def list_all(self) -> list[dict]:
        """List all survey states."""
        with self._lock:
            return list(self._states.values())

    def delete(self, survey_id: str) -> bool:
        """Delete a survey state."""
        with self._lock:
            if survey_id in self._states:
                del self._states[survey_id]
                fpath = os.path.join(self._persist_dir, f"{survey_id}.json")
                if os.path.exists(fpath):
                    os.remove(fpath)
                return True
        return False


survey_store = SurveyStore()

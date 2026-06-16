"""
LangGraph conditional edge routing functions.
"""
from core.state import SurveyState, GapType


def should_run_gap(state: SurveyState) -> str:
    """Route to gap_node if gap_type is selected, skip otherwise."""
    if state.gap_type == GapType.EMPTY:
        return "skip_gap"
    return "run_gap"

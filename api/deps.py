"""
FastAPI dependency injection.
"""
from storage.survey_store import SurveyStore, survey_store
from graph.builder import build_graph
from core.llm_client import LLMClient


def get_survey_store() -> SurveyStore:
    return survey_store


def get_graph():
    return build_graph()


def get_llm_client() -> LLMClient:
    return LLMClient()

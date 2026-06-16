"""Citation Accuracy benchmark for SurveyFlow.

Loads .env from the project root on import so LLMClient, the
Semantic Scholar verifier, and the faithfulness judge can pick up
API keys configured in the .env file.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    # The .env lives at <repo>/survey_agent/.env, i.e. two directories
    # up from this __init__.py.
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    # python-dotenv is not installed; the user is expected to export
    # API keys in their shell before running the benchmark.
    pass

"""
Constants and default values for the survey pipeline system.
"""
import os
from enum import Enum


class GapType(str, Enum):
    THEORETICAL = "Theoretical Gap"
    METHODOLOGICAL = "Methodological Gap"
    DATA = "Data Gap"
    APPLICATION = "Application Gap"
    EVALUATION = "Evaluation Gap"
    PERSPECTIVE = "Perspective Gap"
    EMPTY = ""


GAP_TYPE_DESCRIPTIONS = {
    GapType.THEORETICAL: (
        "Theoretical gaps involve missing foundational frameworks, axioms, or provable guarantees "
        "in the field's understanding of core principles."
    ),
    GapType.METHODOLOGICAL: (
        "Methodological gaps involve lacking systematic approaches, evaluation protocols, or "
        "reproducible experimental frameworks."
    ),
    GapType.DATA: (
        "Data gaps involve insufficient datasets, biased data, lack of benchmark corpora, "
        "or missing real-world data for validation."
    ),
    GapType.APPLICATION: (
        "Application gaps involve underexplored domains, lack of domain-specific adaptations, "
        "or missing practical use cases."
    ),
    GapType.EVALUATION: (
        "Evaluation gaps involve missing metrics, insufficient benchmarking standards, "
        "or lack of comprehensive assessment frameworks."
    ),
    GapType.PERSPECTIVE: (
        "Perspective gaps involve lacking interdisciplinary views, missing comparative analyses "
        "across related fields, or narrow viewpoints."
    ),
    GapType.EMPTY: "No specific gap type selected.",
}


# Default configuration values
DEFAULT_PAPER_COUNT = 20
DEFAULT_YEAR_START = "2020-01-01"
DEFAULT_YEAR_END = ""
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 30
DEFAULT_N_RESULTS = 5
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_RETRIES = 10
DEFAULT_BACKOFF_FACTOR = 1.0
DEFAULT_MAX_WORKERS = 4

# Directory paths
DEFAULT_DATA_ROOT = "./data"
DEFAULT_PDF_PATH = "./data/pdf"
DEFAULT_MD_PATH = "./data/md"
DEFAULT_TSV_PATH = "./data/tsv"
DEFAULT_TXT_PATH = "./data/txt"
DEFAULT_CHROMA_PATH = "./data/chroma"
DEFAULT_ARXIV_PATH = "./data/arxiv_downloads"


def survey_subdir(survey_id: str, base_dir: str) -> str:
    """Return ``base_dir/{survey_id}`` and fail fast if survey_id is empty.

    Several nodes compute paths like
    ``os.path.join(DEFAULT_ARXIV_PATH, survey_id)`` to give every
    run its own subdirectory. If ``survey_id`` is empty (e.g. a
    caller built a ``SurveyState`` without setting it), ``os.path.join``
    silently collapses to ``base_dir`` itself, and the run's PDFs,
    txt JSONs, etc. flatten into the parent folder, mixed with files
    from other runs.

    This helper guards the collapse by raising a clear ``ValueError``
    that points to the root cause. Use it from every node that builds
    a per-run subdirectory.
    """
    if not survey_id or not survey_id.strip():
        raise ValueError(
            "survey_id is empty. SurveyState.survey_id defaults to '' "
            "in core/state.py; the API/CLI entry points and the GQE "
            "benchmark's _run_fresh() set it via uuid4().hex. If you "
            "are calling a graph node directly, set state.survey_id "
            "first."
        )
    return os.path.join(base_dir, survey_id)

# LLM settings
DEFAULT_MODEL = "glm-4-flash"
DEFAULT_CONTEXT_WINDOW = 128000
DEFAULT_MAX_TOKENS = 32768
DEFAULT_SAFETY_MARGIN = 512
DEFAULT_MIN_COMPLETION = 256

# Gap analysis
GAP_ANALYSIS_QUERIES = {
    GapType.THEORETICAL: [
        "What theoretical limitations or open problems are discussed?",
        "What foundational assumptions are not addressed?",
        "What mathematical frameworks are missing?",
    ],
    GapType.METHODOLOGICAL: [
        "What methodological weaknesses or flaws exist?",
        "What experimental designs have limitations?",
        "What validation approaches are missing?",
    ],
    GapType.DATA: [
        "What data limitations are mentioned?",
        "What datasets are missing or insufficient?",
        "What biases or coverage gaps exist?",
    ],
    GapType.APPLICATION: [
        "What real-world applications are not explored?",
        "What domain-specific challenges are unaddressed?",
        "What practical scenarios lack solutions?",
    ],
    GapType.EVALUATION: [
        "What evaluation metrics are missing?",
        "What benchmarking standards are absent?",
        "What comparative analyses are lacking?",
    ],
    GapType.PERSPECTIVE: [
        "What interdisciplinary connections are missing?",
        "What alternative viewpoints are not considered?",
        "What related fields have unexplored connections?",
    ],
}

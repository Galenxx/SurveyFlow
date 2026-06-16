"""Configuration for the Citation Accuracy benchmark.

Defines:
  - The 10 topic grid (one or two per field) to keep the benchmark
    sensitive to broad coverage and to give enough cells for paired
    statistics.
  - The three systems on the ablation ladder: LM_ONLY (parametric
    knowledge only) -> NAIVE_RAG (flat whole-corpus retrieval) ->
    SURVEYFLOW (full cluster-scoped retrieval pipeline).
  - The repetition (seed) dimension: each (system, topic) cell is run
    once per seed in SEEDS.
  - Pipeline-level constants (citation count, paths, thresholds, etc.).

The grid is intentionally different from `evaluation/configs/topics.py`
(GQE benchmark): this benchmark focuses on citation accuracy and
wants a fresh, broader field coverage.
"""

from __future__ import annotations
import os
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = BENCH_ROOT / "results"
SURVEY_OUTPUT_DIR = BENCH_ROOT / "generated_surveys"

# Back-compat aliases (some legacy modules still import these names).
SURVEYFLOW_RESULTS = RESULTS_ROOT / "surveyflow"
LLM_ONLY_RESULTS = RESULTS_ROOT / "llm_only"


def results_dir(system_id: str) -> Path:
    """Per-system results directory, derived from the system id.

    Replaces the old hard-coded ``SURVEYFLOW_RESULTS`` / ``LLM_ONLY_RESULTS``
    pair so a third (or Nth) system needs no special-casing. The directory
    is created on first access.
    """
    d = RESULTS_ROOT / system_id.lower()
    d.mkdir(parents=True, exist_ok=True)
    return d


for _p in (SURVEYFLOW_RESULTS, LLM_ONLY_RESULTS, SURVEY_OUTPUT_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# --- Repetition (seed) dimension ----------------------------------------
#
# Each (system, topic) cell is run once per seed so the benchmark can
# report variance, run paired significance tests, and average out the
# stochasticity of the generator/judge LLMs. The seed value is threaded
# into result filenames as ``{topic_id}__seed{k}__{phase}.json``.
SEEDS: list[int] = [0, 1, 2]


# --- Topic grid ----------------------------------------------------------

TOPICS: list[dict] = [
    {
        "id": "code_review_llm",
        "label": "LLM-based Automated Code Review",
        "topic": (
            "Large Language Models for Automated Code Review: "
            "Techniques, Benchmarks, and Industrial Deployment"
        ),
        "field": "Software Engineering / NLP",
    },
    {
        "id": "diffusion_3d",
        "label": "Diffusion Models for 3D Content Generation",
        "topic": (
            "Diffusion Models for 3D Shape and Scene Generation: "
            "Methods, Representations, and Evaluation"
        ),
        "field": "Computer Vision / Graphics",
    },
    {
        "id": "gnn_drug",
        "label": "Graph Neural Networks for Drug Discovery",
        "topic": (
            "Graph Neural Networks for Drug Discovery and Molecular "
            "Property Prediction"
        ),
        "field": "Computational Biology / ML",
    },
    {
        "id": "short_video_misinfo",
        "label": "Misinformation Detection on Short Video Platforms",
        "topic": (
            "Detection of Misinformation and Harmful Content on "
            "Short-Form Video Platforms: Methods, Datasets, and "
            "Multimodal Approaches"
        ),
        "field": "Social Computing / Multimedia",
    },
    {
        "id": "affective_remote_work",
        "label": "Affective Computing in Remote Workplaces",
        "topic": (
            "Affective Computing for Remote Workplace Monitoring: "
            "Sensing, Inference, and Ethical Considerations"
        ),
        "field": "HCI / Organizational Behavior",
    },
    {
        "id": "federated_health",
        "label": "Federated Learning for Healthcare",
        "topic": (
            "Federated Learning for Privacy-Preserving Healthcare: "
            "Architectures, Communication Efficiency, and Clinical "
            "Deployment"
        ),
        "field": "Machine Learning / Health Informatics",
    },
    {
        "id": "neural_audio_codec",
        "label": "Neural Audio Codecs",
        "topic": (
            "Neural Audio Codecs and Discrete Speech Representations: "
            "Architectures, Tokenization, and Generative Applications"
        ),
        "field": "Speech / Signal Processing",
    },
    {
        "id": "llm_agent_planning",
        "label": "Planning in LLM Agents",
        "topic": (
            "Planning and Reasoning in Large Language Model Agents: "
            "Methods, Tool Use, and Evaluation Benchmarks"
        ),
        "field": "NLP / Autonomous Agents",
    },
    {
        "id": "battery_soh_ml",
        "label": "ML for Battery State-of-Health",
        "topic": (
            "Machine Learning for Lithium-Ion Battery State-of-Health "
            "Estimation and Remaining-Useful-Life Prediction"
        ),
        "field": "Energy Systems / ML",
    },
    {
        "id": "txt2sql",
        "label": "Text-to-SQL with LLMs",
        "topic": (
            "Text-to-SQL Generation with Large Language Models: "
            "Schema Linking, Benchmarks, and Robustness"
        ),
        "field": "Databases / NLP",
    },
]


def topic_by_id(topic_id: str) -> dict:
    for t in TOPICS:
        if t["id"] == topic_id:
            return t
    raise KeyError(topic_id)


# --- Systems under comparison --------------------------------------------

SYSTEMS: list[dict] = [
    {
        "id": "SURVEYFLOW",
        "label": "SurveyFlow (cluster-scoped retrieval)",
        "description": (
            "Full SurveyFlow pipeline: retrieves and parses real "
            "PDFs, clusters them, and generates each section with "
            "citation grounding scoped to its cluster's evidence. "
            "Top of the ablation ladder."
        ),
    },
    {
        "id": "NAIVE_RAG",
        "label": "Naive RAG (flat whole-corpus retrieval)",
        "description": (
            "Same downloaded corpus and same generator LLM as "
            "SurveyFlow, but with the clustering / outline structure "
            "ablated away: each section is grounded by a single flat "
            "retrieval over the WHOLE corpus, and references are "
            "assembled from whatever the retriever surfaced. Isolates "
            "the contribution of SurveyFlow's structure from that of "
            "retrieval alone (middle rung of the ablation ladder)."
        ),
    },
    {
        "id": "LLM_ONLY",
        "label": "Single-shot LLM (no retrieval)",
        "description": (
            "Same generator LLM (glm-5.1), but with NO retrieval "
            "context. The model is asked to produce a survey from "
            "parametric knowledge only, and is told to cite ~10 "
            "real papers. This is the hallucination-prone baseline "
            "and the bottom rung of the ablation ladder."
        ),
    },
]


def system_by_id(system_id: str) -> dict:
    for s in SYSTEMS:
        if s["id"] == system_id:
            return s
    raise KeyError(system_id)


# --- Pipeline-level constants -------------------------------------------

CITATIONS_PER_SURVEY = 10
EXISTENCE_SIM_THRESHOLD = 0.85
IDENTIFIER_SIM_THRESHOLD = 0.80

# Generation guards (Task #5). A survey that comes back with far fewer
# references than requested is a red flag (the old hard-coded 3-cluster
# bug used to silently drop whole sections, leaving 2-3 refs). The
# generate phase records the corpus stats and the realized reference
# count, and emits a WARNING into the generate.json when the realized
# count falls below CITATION_COUNT_WARN_FRACTION * requested. It does
# NOT abort — a low count is itself a datum worth recording — but it is
# surfaced loudly instead of failing silently.
CITATION_COUNT_WARN_FRACTION = 0.6

# Model used by LLM_ONLY generator (must match SurveyFlow's generator
# model for a fair apples-to-apples comparison).
LLM_ONLY_MODEL = "glm-5.1"

# Faithfulness judge uses the opposite-family model to mitigate
# self-preference bias (same convention as the GQE benchmark). If
# the preferred model is unavailable (no API key), the runner
# transparently falls back to JUDGE_FALLBACK_MODEL.
JUDGE_MODEL = "deepseek-v4-pro"
JUDGE_FALLBACK_MODEL = "glm-5.1"
JUDGE_TEMPERATURE = 0.0

# arXiv / Semantic Scholar rate-limits.
ARXIV_API_BASE = "http://export.arxiv.org/api/query"
S2_API_BASE = "https://api.semanticscholar.org/graph/v1/paper"
CROSSREF_API_BASE = "https://api.crossref.org/works"
HTTP_TIMEOUT = 15.0
# The project's .env uses SEMANTIC_SCHOLAR_API_KEY; fall back to
# S2_API_KEY for users who set it directly in their shell.
S2_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY", "")

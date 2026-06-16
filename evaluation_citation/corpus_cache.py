"""Per-topic corpus cache — ingest ONCE, reuse across all systems and seeds.

Why this exists
---------------
The benchmark grid is 3 systems x 10 topics x 3 seeds = 90 cells. Two of the
three systems (SURVEYFLOW, NAIVE_RAG) need a real corpus: they download PDFs,
parse + chunk them, and embed the chunks into ChromaDB before any writing
happens. That ingestion (query -> download -> splitter) is BY FAR the slowest
part of a cell (~hundreds of seconds: arXiv/S2 search, PDF download with up to
5 retry cycles, PDF parsing, embedding).

In the original design every cell re-ran ingestion from scratch under a random
``survey_id`` (uuid4), so the SAME topic's corpus was downloaded and embedded
**6 times** (SURVEYFLOW x3 seeds + NAIVE_RAG x3 seeds). That is the dominant
cost and it is pure waste — the corpus for a topic does not depend on the
system or the seed.

This module ingests each topic exactly ONCE under a STABLE, topic-derived
``survey_id`` and caches the resulting ingestion state to disk. Every cell then
loads that cached state and runs only the writing half (retriever -> ... ->
writer for SURVEYFLOW, flat-retrieve + write for NAIVE_RAG). Ingestion drops
from 60 runs to 10.

Methodological bonus
--------------------
Fixing the corpus per topic also makes the SURVEYFLOW-vs-NAIVE_RAG ablation
cleaner: both systems now compete over the EXACT same chunks, so the measured
difference is attributable purely to retrieval *structure* (cluster-scoped vs
flat), not to the two systems happening to download different papers. The only
remaining stochasticity is the writer LLM's sampling (temperature), which is
exactly the replicate dimension the seeds are meant to capture.

Stability of the cache key
--------------------------
The ``survey_id`` is derived deterministically from the topic id
(``corpus_<topic_id>``), NOT a random uuid. This means:
  * the on-disk PDF dir / txt dir / ChromaDB collections for a topic are stable
    across runs and sessions, so a re-run transparently hits the existing
    corpus instead of re-downloading;
  * the cache cannot collide with the historical uuid-keyed dirs left by the
    old design.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from evaluation_citation.configs import BENCH_ROOT, topic_by_id

CORPUS_CACHE_DIR = BENCH_ROOT / "corpus_cache"
CORPUS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Fields produced by the ingestion half (query -> download -> splitter) that
# the writing half depends on. We persist exactly these so a SurveyState can
# be rebuilt for any system/seed without touching the network again.
_INGEST_FIELDS = (
    # query node
    "arxiv_query",
    "generic_arxiv_query",
    "s2_query",
    "s2_generic_query",
    # download node
    "survey_id",
    "tsv_path",
    "paper_list",
    "download_status",
    "failed_papers",
    # splitter node
    "chroma_collections",
    "total_chunks",
    "embedding_status",
    "citation_paper_meta",
)


def stable_survey_id(topic_id: str) -> str:
    """Deterministic, collision-free survey_id for a topic's shared corpus.

    Stable across runs/sessions (so re-runs reuse the on-disk corpus) and
    distinct from the historical uuid4 ids (so no dir collision).
    """
    return f"corpus_{topic_id}"


def _cache_path(topic_id: str) -> Path:
    return CORPUS_CACHE_DIR / f"{topic_id}.json"


def corpus_exists(topic_id: str) -> bool:
    p = _cache_path(topic_id)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    # A usable corpus must have at least one embedded collection.
    return bool(data.get("chroma_collections"))


def _run_ingestion(topic_id: str, paper_count: int) -> dict:
    """Run query -> download -> splitter ONCE under the stable survey_id and
    return the ingestion-field subset of the resulting state as a dict.

    Raises RuntimeError if any ingestion node fails.
    """
    from core.state import SurveyState, GapType
    from graph.nodes import query_node, download_node, splitter_node
    from storage.survey_store import survey_store

    topic = topic_by_id(topic_id)
    state = SurveyState(
        user_input=topic["topic"],
        gap_type=GapType.EMPTY,        # no gap branch — keep ingestion lean
        paper_count=paper_count,
        enable_citations=True,
        survey_id=stable_survey_id(topic_id),
    )
    survey_store.update(state.survey_id, state.model_dump())

    state = query_node(state)
    if str(getattr(state, "status", "")).endswith("FAILED"):
        raise RuntimeError(f"ingestion failed in query_node: {state.error_message}")
    state = download_node(state)
    if str(getattr(state, "status", "")).endswith("FAILED"):
        raise RuntimeError(f"ingestion failed in download_node: {state.error_message}")
    state = splitter_node(state)
    if str(getattr(state, "status", "")).endswith("FAILED"):
        raise RuntimeError(f"ingestion failed in splitter_node: {state.error_message}")

    survey_store.update(state.survey_id, state.model_dump())

    full = state.model_dump()
    return {k: full.get(k) for k in _INGEST_FIELDS}


def ensure_corpus(topic_id: str, *, paper_count: int, force: bool = False) -> dict:
    """Return the cached ingestion state for a topic, ingesting it if absent.

    The returned dict holds exactly the ``_INGEST_FIELDS`` and is enough to
    rebuild a SurveyState for the writing half. Idempotent: if a valid cache
    file already exists (and ``force`` is False) it is loaded from disk and NO
    network/embedding work happens.
    """
    p = _cache_path(topic_id)
    if not force and corpus_exists(topic_id):
        return json.loads(p.read_text(encoding="utf-8"))

    ingest = _run_ingestion(topic_id, paper_count)
    # Provenance: record what the corpus looks like for later inspection.
    ingest["_corpus_stats"] = {
        "topic_id": topic_id,
        "survey_id": ingest.get("survey_id", ""),
        "n_papers_in_list": len(ingest.get("paper_list") or []),
        "n_collections": len(ingest.get("chroma_collections") or []),
        "total_chunks": ingest.get("total_chunks", 0),
        "n_citation_meta": len(ingest.get("citation_paper_meta") or {}),
    }
    p.write_text(json.dumps(ingest, indent=2, ensure_ascii=False), encoding="utf-8")
    return ingest


def load_state_from_corpus(topic_id: str, *, paper_count: int, gap_type=None):
    """Build a fresh SurveyState pre-populated with a topic's cached corpus.

    The corpus must already be cached (call :func:`ensure_corpus` first). The
    returned state is ready for the writing half: it carries paper_list,
    chroma_collections, citation_paper_meta, tsv_path, etc., under the topic's
    stable survey_id, so retriever/writer nodes find the embedded chunks and
    the on-disk TSV without any new ingestion.
    """
    from core.state import SurveyState, GapType

    p = _cache_path(topic_id)
    if not corpus_exists(topic_id):
        raise FileNotFoundError(
            f"no cached corpus for topic '{topic_id}'. Call ensure_corpus() first."
        )
    ingest = json.loads(p.read_text(encoding="utf-8"))

    topic = topic_by_id(topic_id)
    state = SurveyState(
        user_input=topic["topic"],
        gap_type=gap_type if gap_type is not None else GapType.EMPTY,
        paper_count=paper_count,
        enable_citations=True,
        survey_id=ingest.get("survey_id") or stable_survey_id(topic_id),
    )
    for k in _INGEST_FIELDS:
        if k == "survey_id":
            continue  # already set above
        if k in ingest and ingest[k] is not None:
            setattr(state, k, ingest[k])
    return state

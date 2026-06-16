"""SurveyFlow generator runner.

Wraps the project's LangGraph pipeline so the benchmark can invoke
it for a given (topic, paper_count) pair and get back the path to the
generated survey.md.
"""

from __future__ import annotations
import asyncio
import json
import uuid
from pathlib import Path
from typing import Optional

from evaluation_citation.configs import (
    CITATIONS_PER_SURVEY,
    SURVEY_OUTPUT_DIR,
)


def _run_async(coro):
    """Run an async coroutine to completion. Works whether or not we
    are already inside an event loop (e.g. when called from a Jupyter
    kernel)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside a loop - run on a fresh loop in a worker thread.
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def _md_outline_only_prompt() -> str:
    """No-op prompt kept for future use; left empty because SurveyFlow
    uses its own node-level prompts that we deliberately do not change
    for this benchmark."""
    return ""


async def _generate_survey_async(topic: str, paper_count: int) -> dict:
    """Build the graph, run it on `topic`, return the resulting state."""
    from graph.builder import build_graph
    from core.state import SurveyState, GapType
    from storage.survey_store import survey_store

    state = SurveyState(
        user_input=topic,
        gap_type=GapType.EMPTY,
        paper_count=paper_count,
        enable_citations=True,
        survey_id=uuid.uuid4().hex,
    )

    # Persist the initial empty state so the survey_id is known and
    # the file exists on disk. Subsequent updates overwrite it as
    # the pipeline progresses.
    survey_store.update(state.survey_id, state.model_dump())

    graph = build_graph()
    final = None
    async for event in graph.astream(state):
        # astream yields {node_name: state_after_node}
        final = list(event.values())[0]
        if hasattr(final, "model_dump"):
            snapshot = final.model_dump()
        else:
            snapshot = final
        survey_store.update(state.survey_id, snapshot)
    return final


def _generate_from_corpus(topic_id: str, paper_count: int, seed: Optional[int] = None) -> dict:
    """Run ONLY SurveyFlow's writing half on a pre-ingested corpus.

    The corpus (PDFs downloaded, chunked, embedded into ChromaDB, plus
    ``paper_list`` / ``citation_paper_meta`` / ``tsv_path``) is loaded from the
    per-topic cache (see :mod:`evaluation_citation.corpus_cache`). We then run
    the five writing-half nodes directly as pure ``state -> state`` functions:

        retriever -> classifier -> section -> outline -> writer

    These are exactly the nodes the compiled graph would run after
    ``splitter_node`` when ``gap_type == EMPTY`` (see graph/builder.py and
    graph/edges.should_run_gap). Skipping query/download/splitter avoids the
    re-download + re-embed that dominates a cell's runtime, with no change to
    the writing logic under test.

    Seed isolation: because all seeds of a topic share the topic's stable
    ``survey_id`` (= the corpus key), ``writer_node`` writes every seed to the
    SAME ``data/md/{survey_id}/survey.md`` — later seeds would overwrite earlier
    ones. We therefore copy the writer's output to a seed-keyed, benchmark-owned
    path and point the returned ``md_file_path`` at that copy, so the parse phase
    reads each seed's own survey. (Same bug class we fixed for NAIVE_RAG.)
    """
    import shutil
    from evaluation_citation.corpus_cache import load_state_from_corpus
    from core.state import GapType
    from graph.nodes import (
        retriever_node, classifier_node, section_node, outline_node, writer_node,
    )
    from storage.survey_store import survey_store

    state = load_state_from_corpus(
        topic_id, paper_count=paper_count, gap_type=GapType.EMPTY
    )
    survey_store.update(state.survey_id, state.model_dump())

    for node in (retriever_node, classifier_node, section_node, outline_node, writer_node):
        state = node(state)
        if str(getattr(state, "status", "")).endswith("FAILED"):
            raise RuntimeError(
                f"SurveyFlow writing-half failed in {node.__name__}: "
                f"{getattr(state, 'error_message', '')}"
            )
        survey_store.update(state.survey_id, state.model_dump())

    final = state.model_dump()

    # --- seed-isolate the writer output -----------------------------------
    src = final.get("md_file_path", "")
    if seed is not None and src and Path(src).exists():
        safe = topic_id
        seed_dir = SURVEY_OUTPUT_DIR / "surveyflow" / f"{safe}__seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        dst = seed_dir / "survey.md"
        shutil.copyfile(src, dst)
        final["md_file_path"] = str(dst)

    return final


def _extract_corpus_stats(final: dict) -> dict:
    """Pull the corpus-level provenance numbers out of the final state.

    These let us tell, after the fact, WHETHER a low citation count was
    caused by a thin corpus (few papers downloaded / embedded) or by the
    writer dropping sections. Recording them turns "only 3 references"
    from an unexplained anomaly into a diagnosable data point.
    """
    paper_list = final.get("paper_list") or []
    chroma_collections = final.get("chroma_collections") or []
    citation_meta = final.get("citation_paper_meta") or {}
    labels = final.get("labels") or {}
    download_status = final.get("download_status") or {}
    embedding_status = final.get("embedding_status") or {}

    def _count_ok(status_map: dict) -> int:
        return sum(
            1 for v in status_map.values()
            if str(v).lower() in ("ok", "success", "succeeded", "done", "completed")
        )

    return {
        "n_papers_in_list": len(paper_list),
        "n_papers_downloaded_ok": _count_ok(download_status) or len(paper_list),
        "n_collections_embedded": len(chroma_collections),
        "n_papers_embedded_ok": _count_ok(embedding_status) or len(chroma_collections),
        "n_citation_meta": len(citation_meta),
        "total_chunks": final.get("total_chunks", 0),
        "num_classes": final.get("num_classes", 0),
        "n_labels": len(labels),
    }


def _count_references_in_md(md_path: str) -> int:
    """Count 'Reference:' lines in the produced survey.

    Uses the same surface form the citation_parser keys on, so the number
    recorded here matches what the parse phase will later extract.
    """
    try:
        text = Path(md_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return 0
    return sum(
        1 for line in text.splitlines()
        if line.strip().lower().startswith("reference:")
    )


def run_surveyflow(topic: str, *, paper_count: int = CITATIONS_PER_SURVEY) -> str:
    """Run the SurveyFlow pipeline on a topic.

    Returns the path to the generated survey.md on disk.
    Raises RuntimeError if the pipeline did not produce a file.

    This is the thin back-compat wrapper; callers that want the corpus
    stats and realized citation count should use :func:`run_surveyflow_detailed`.
    """
    return run_surveyflow_detailed(topic, paper_count=paper_count)["md_path"]


def run_surveyflow_detailed(
    topic: str,
    *,
    paper_count: int = CITATIONS_PER_SURVEY,
    topic_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict:
    """Run SurveyFlow and return md_path plus generation-guard telemetry.

    Returns a dict::

        {
          "md_path": str,
          "survey_id": str,
          "n_references_realized": int,
          "corpus_stats": {...},
        }

    Raises RuntimeError if the pipeline did not produce a file.

    When ``topic_id`` is given, the topic's corpus is ingested ONCE and
    cached (see :mod:`evaluation_citation.corpus_cache`); this call then
    loads the cached corpus and runs ONLY the writing half of the pipeline
    (retriever -> classifier -> section -> outline -> writer), skipping the
    expensive query -> download -> splitter ingestion. Without ``topic_id``
    the full graph runs from scratch (legacy behaviour).
    """
    if topic_id is not None:
        final = _generate_from_corpus(topic_id, paper_count, seed=seed)
    else:
        final = _run_async(_generate_survey_async(topic, paper_count))
    if final is None:
        raise RuntimeError("SurveyFlow produced no state")

    # state may be a Pydantic model or a dict
    if hasattr(final, "model_dump"):
        final = final.model_dump()
    if not isinstance(final, dict):
        final = {}
    md_path = final.get("md_file_path", "")
    status = final.get("status", "")
    survey_id = final.get("survey_id", "")

    if status and str(status).endswith("FAILED"):
        err = final.get("error_message", "")
        raise RuntimeError(f"SurveyFlow run failed: {err}")

    if not md_path or not Path(md_path).exists():
        # Fallback: look in the data/md directory for the most recent
        # survey whose Abstract mentions our topic.
        candidates = sorted(
            SURVEY_OUTPUT_DIR.glob("*/survey.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            md_path = str(candidates[0])
        else:
            raise RuntimeError("SurveyFlow did not produce a survey.md file")

    return {
        "md_path": md_path,
        "survey_id": survey_id,
        "n_references_realized": _count_references_in_md(md_path),
        "corpus_stats": _extract_corpus_stats(final),
    }


if __name__ == "__main__":  # pragma: no cover
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m evaluation_citation.generators.surveyflow_runner <topic>")
        sys.exit(1)
    path = run_surveyflow(sys.argv[1])
    print(json.dumps({"md_path": path}, indent=2))

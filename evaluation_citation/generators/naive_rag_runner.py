"""NAIVE_RAG generator runner — the middle rung of the ablation ladder.

NAIVE_RAG isolates the contribution of SurveyFlow's *structure*
(clustering + per-cluster outline + cluster-scoped retrieval) from the
contribution of *retrieval itself*. It therefore:

  * REUSES SurveyFlow's real ingestion — the same query/download/splitter
    nodes — so the corpus (downloaded PDFs, chunked + embedded into
    ChromaDB, and ``citation_paper_meta``) is byte-for-byte the same
    process the full system uses. No second download path to drift out
    of sync.

  * ABLATES the structure — instead of LM-decided clusters and a
    cluster-scoped outline, it writes a FIXED, generic survey outline
    (Background / Methods / Applications / Challenges). Each section is
    grounded by a SINGLE FLAT retrieval over the WHOLE corpus (every
    collection), not a cluster-scoped one.

  * KEEPS the writing surface identical — it reuses SurveyFlow's exact
    ``SECTION_WRITING_USER_TEMPLATE`` (APA source labels + in-text
    citation rules) and the same APA reference-line format, so any
    citation-quality difference is attributable to structure/retrieval
    scope, not to a different prompt.

The result is written to a fixed, benchmark-owned location
(``generated_surveys/naive_rag/<safe_topic>__seed<k>/survey.md``) — never
the random ``survey_id`` directory — so the artifact can never be lost.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path
from typing import Optional

from evaluation_citation.configs import (
    CITATIONS_PER_SURVEY,
    SURVEY_OUTPUT_DIR,
)


# Fixed, topic-agnostic outline. These titles are deliberately generic so
# that NAIVE_RAG carries NONE of SurveyFlow's LM-decided cluster structure.
# (level, title) mirrors SurveyFlow's outline shape; the numbering matters
# only for human readability here since retrieval is flat.
_FIXED_OUTLINE: list[tuple[int, str]] = [
    (1, "1 Abstract"),
    (1, "2 Introduction"),
    (1, "3 Background and Foundations"),
    (1, "4 Methods and Techniques"),
    (1, "5 Applications and Systems"),
    (1, "6 Challenges and Open Problems"),
    (1, "7 Future Directions"),
    (1, "8 Conclusion"),
]

# The four body sections that get retrieval-grounded content (everything
# between Introduction and Future Directions).
_BODY_SECTION_TITLES = [
    "3 Background and Foundations",
    "4 Methods and Techniques",
    "5 Applications and Systems",
    "6 Challenges and Open Problems",
]


def _run_async(coro):
    """Run an async coroutine to completion whether or not we are already
    inside an event loop (mirrors surveyflow_runner._run_async)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


async def _ingest_async(topic: str, paper_count: int):
    """Run ONLY the ingestion half of SurveyFlow (query -> download ->
    splitter) so we get the exact same corpus + citation_paper_meta the
    full system would, then hand the state back for flat writing.

    We invoke the three node functions directly rather than the compiled
    graph: the graph would carry on into classifier/section/outline/writer
    (the very structure we are ablating), and stopping it mid-stream is
    messier than just calling the three pure ``state -> state`` nodes.
    """
    from core.state import SurveyState, GapType
    from graph.nodes import query_node, download_node, splitter_node
    from storage.survey_store import survey_store

    state = SurveyState(
        user_input=topic,
        gap_type=GapType.EMPTY,          # no gap branch — keep ingestion lean
        paper_count=paper_count,
        enable_citations=True,
        survey_id=uuid.uuid4().hex,
    )
    survey_store.update(state.survey_id, state.model_dump())

    # query -> download -> splitter. download_node already embeds; splitter
    # reuses that work and builds citation_paper_meta.
    state = query_node(state)
    if str(getattr(state, "status", "")).endswith("FAILED"):
        raise RuntimeError(f"NAIVE_RAG ingestion failed in query_node: {state.error_message}")
    state = download_node(state)
    if str(getattr(state, "status", "")).endswith("FAILED"):
        raise RuntimeError(f"NAIVE_RAG ingestion failed in download_node: {state.error_message}")
    state = splitter_node(state)
    if str(getattr(state, "status", "")).endswith("FAILED"):
        raise RuntimeError(f"NAIVE_RAG ingestion failed in splitter_node: {state.error_message}")

    survey_store.update(state.survey_id, state.model_dump())
    return state


def _flat_retrieve(
    section_title: str,
    topic: str,
    state,
    embedder,
    chroma,
    paper_map: dict,
    used_collections: set,
) -> str:
    """Retrieve context for one section over the WHOLE corpus (flat).

    This is the deliberate ablation of SurveyFlow's cluster-scoped
    retrieval: instead of restricting the query to one cluster's
    collections, we query EVERY collection and keep the closest chunks.
    Records which collections actually surfaced chunks (in
    ``used_collections``) so references can be assembled from real hits.
    """
    from evaluation_citation.generators._writing import make_source_label

    collections = list(state.chroma_collections or [])
    if not collections:
        return ""

    query_text = f"{topic} — {section_title}"
    try:
        q_emb = embedder.embed_query(query_text)
    except Exception as e:
        state.add_log(f"NAIVE_RAG: embedding failed for '{section_title}' — {e}")
        return ""

    scored: list[tuple[float, str]] = []  # (distance, labelled_chunk)
    for coll in collections:
        try:
            result = chroma.query(collection_name=coll, query_embeddings=[q_emb], n_results=5)
        except Exception as e:
            state.add_log(f"NAIVE_RAG: query failed for '{coll}' — {type(e).__name__}: {e}")
            continue
        if not isinstance(result, dict) or not result.get("documents"):
            continue
        docs = result["documents"][0] if result["documents"] else []
        metas = result.get("metadatas") or [[]]
        meta_list = metas[0] if metas else []
        dists = result.get("distances") or [[]]
        dist_list = dists[0] if dists else []
        for i, doc_text in enumerate(docs):
            chunk_index = i
            if meta_list and i < len(meta_list) and isinstance(meta_list[i], dict):
                chunk_index = meta_list[i].get("chunk_index", i)
            dist = dist_list[i] if i < len(dist_list) and dist_list[i] is not None else 1.0
            label = make_source_label(coll, paper_map, chunk_index)
            scored.append((dist, f"[{label}]\n{doc_text}"))
            used_collections.add(coll)

    # Keep the globally-closest chunks across the whole corpus. The chunk
    # count (8) and character budget (3500) are kept IDENTICAL to
    # SurveyFlow's cluster-scoped retrieval (writer_node._retrieve_context)
    # so the two systems receive the same evidence budget and the only
    # difference is HOW that evidence is organized (flat whole-corpus here
    # vs cluster-scoped there) — a clean ablation of structure, not volume.
    scored.sort(key=lambda x: x[0])
    top = [chunk for _, chunk in scored[:8]]
    combined = "\n\n".join(top)
    return combined[:3500]


def _write_section(llm, section_title: str, context: str) -> str:
    from prompts.writer_prompts import (
        SECTION_WRITING_SYSTEM_PROMPT,
        SECTION_WRITING_USER_TEMPLATE,
    )
    prompt = SECTION_WRITING_USER_TEMPLATE.format(
        section_title=section_title,
        context=context,
    )
    return llm.generate(
        system_prompt=SECTION_WRITING_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=1024,
        temperature=0.5,
        model="deepseek-v4-pro",
    )


def _write_macro(llm, kind: str, topic: str, body_text: str) -> str:
    """Generate Abstract / Introduction / Conclusion / Future Directions
    from the assembled body, reusing SurveyFlow's macro prompts so the two
    systems differ only in structure, not in macro-section prompting."""
    from prompts.writer_prompts import (
        ABSTRACT_SYSTEM_PROMPT, ABSTRACT_USER_TEMPLATE,
        INTRODUCTION_SYSTEM_PROMPT, INTRODUCTION_USER_TEMPLATE,
        CONCLUSION_SYSTEM_PROMPT, CONCLUSION_USER_TEMPLATE,
        FUTURE_WORK_SYSTEM_PROMPT, FUTURE_WORK_USER_TEMPLATE,
    )
    if kind == "abstract":
        sys_p, user_p = ABSTRACT_SYSTEM_PROMPT, ABSTRACT_USER_TEMPLATE.format(context=body_text)
        marker = "Abstract:"
    elif kind == "introduction":
        sys_p, user_p = INTRODUCTION_SYSTEM_PROMPT, INTRODUCTION_USER_TEMPLATE.format(title=topic, context=body_text)
        marker = "Introduction:"
    elif kind == "conclusion":
        sys_p, user_p = CONCLUSION_SYSTEM_PROMPT, CONCLUSION_USER_TEMPLATE.format(context=body_text)
        marker = "Conclusion:"
    elif kind == "future":
        sys_p, user_p = FUTURE_WORK_SYSTEM_PROMPT, FUTURE_WORK_USER_TEMPLATE.format(
            body_text=body_text, gap_report="Not available"
        )
        marker = "Future Directions:"
    else:
        raise ValueError(kind)

    resp = llm.generate(
        system_prompt=sys_p,
        user_prompt=user_p,
        max_tokens=2048 if kind == "introduction" else 1024,
        temperature=0.5,
        model="deepseek-v4-pro",
    )
    idx = resp.find(marker)
    if idx != -1:
        return resp[idx + len(marker):].strip()
    return resp.strip()


def run_naive_rag(
    topic: str,
    *,
    paper_count: int = CITATIONS_PER_SURVEY,
    out_dir: Optional[Path] = None,
    seed: Optional[int] = None,
) -> str:
    """Thin wrapper returning just the md_path (back-compat)."""
    return run_naive_rag_detailed(
        topic, paper_count=paper_count, out_dir=out_dir, seed=seed
    )["md_path"]


def run_naive_rag_detailed(
    topic: str,
    *,
    paper_count: int = CITATIONS_PER_SURVEY,
    out_dir: Optional[Path] = None,
    seed: Optional[int] = None,
    corpus_state=None,
) -> dict:
    """Run the NAIVE_RAG baseline and return md_path + generation-guard
    telemetry, mirroring ``run_surveyflow_detailed``'s contract::

        {
          "md_path": str,
          "survey_id": str,
          "n_references_realized": int,
          "corpus_stats": {...},
        }

    If ``corpus_state`` is provided (a SurveyState already carrying the
    ingested corpus — chroma_collections + citation_paper_meta), the
    ingestion half (query→download→splitter) is skipped entirely and we
    write directly off the shared corpus. This is the cross-cell corpus
    reuse path: one ingestion per topic, reused by every seed and by
    both SURVEYFLOW and NAIVE_RAG, so the two systems compete on the
    EXACT same chunks (cleaner ablation) at a fraction of the cost.
    """
    from core.llm_client import LLMClient
    from core.embedder import get_embedder
    from core.chroma_client import get_chroma_manager
    from evaluation_citation.generators._writing import (
        assemble_references,
        clean_section_content,
        count_reference_lines,
    )

    state = corpus_state if corpus_state is not None else _run_async(
        _ingest_async(topic, paper_count)
    )

    paper_map = dict(state.citation_paper_meta or {})
    llm = LLMClient()
    embedder = get_embedder()
    chroma = get_chroma_manager()

    used_collections: set = set()
    section_content: dict[str, str] = {}
    for title in _BODY_SECTION_TITLES:
        ctx = _flat_retrieve(
            title, topic, state, embedder, chroma, paper_map, used_collections
        )
        if not ctx:
            state.add_log(f"NAIVE_RAG: no context for '{title}' — skipping")
            continue
        section_content[title] = clean_section_content(
            _write_section(llm, title, ctx), section_title=title
        )

    body_text = "\n\n".join(section_content.values())
    abstract = _write_macro(llm, "abstract", topic, body_text)
    introduction = _write_macro(llm, "introduction", topic, body_text)
    conclusion = _write_macro(llm, "conclusion", topic, body_text)
    future = _write_macro(llm, "future", topic, body_text)

    # References come from the collections that actually surfaced chunks —
    # i.e. the papers the writer was really grounded in — exactly mirroring
    # SurveyFlow's "references from used papers" contract.
    references = assemble_references(used_collections, paper_map)

    # --- assemble markdown -------------------------------------------
    macro = {
        "1 Abstract": abstract,
        "2 Introduction": introduction,
        "7 Future Directions": future,
        "8 Conclusion": conclusion,
    }
    parts: list[str] = []
    for level, title in _FIXED_OUTLINE:
        parts.append(f"{'#' * level} {title}\n")
        if title in macro and macro[title]:
            parts.append(macro[title] + "\n")
        elif title in section_content:
            parts.append(section_content[title] + "\n")
    parts.append("# References\n")
    parts.append(references + "\n")
    survey_text = "\n".join(parts)

    # --- persist to a FIXED, benchmark-owned location ----------------
    # The path is keyed by seed so repeated runs of the same topic under
    # different seeds never overwrite each other (each seed is an independent
    # replicate). Without the seed suffix all seeds collapsed onto one
    # survey.md, destroying the seed dimension.
    if out_dir is None:
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", topic[:50]).strip("_")
        if seed is not None:
            safe = f"{safe}__seed{seed}"
        out_dir = SURVEY_OUTPUT_DIR / "naive_rag" / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "survey.md"
    md_path.write_text(survey_text, encoding="utf-8")

    return {
        "md_path": str(md_path),
        "survey_id": state.survey_id,
        "n_references_realized": count_reference_lines(survey_text),
        "corpus_stats": {
            "n_papers_in_list": len(state.paper_list or []),
            "n_collections_embedded": len(state.chroma_collections or []),
            "n_citation_meta": len(paper_map),
            "n_collections_used_in_writing": len(used_collections),
            "total_chunks": getattr(state, "total_chunks", 0),
        },
    }


if __name__ == "__main__":  # pragma: no cover
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m evaluation_citation.generators.naive_rag_runner <topic>")
        sys.exit(1)
    print(run_naive_rag(sys.argv[1]))

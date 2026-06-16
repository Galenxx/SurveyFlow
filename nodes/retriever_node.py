"""
Retriever Node: Generate gap-aware descriptions and citation evidence for each paper.
"""
import concurrent.futures
import json
from pathlib import Path

from core.state import SurveyState, GapType
from core.llm_client import LLMClient
from core.embedder import get_embedder
from core.chroma_client import get_chroma_manager
from core.tsv_manager import TSVManager
from core.constants import DEFAULT_MAX_WORKERS, DEFAULT_N_RESULTS, GAP_TYPE_DESCRIPTIONS
from streaming.manager import get_streaming_manager
from prompts.writer_prompts import (
    SENTENCE_PATTERN_SYSTEM_PROMPT,
    SENTENCE_PATTERN_USER_TEMPLATE,
    GAP_PAPER_DESCRIPTION_SYSTEM_PROMPT,
    GAP_PAPER_DESCRIPTION_USER_TEMPLATE,
)


def retriever_node(state: SurveyState) -> SurveyState:
    """Generate paper descriptions using gap-aware multi-query retrieval + LLM."""
    state.current_node = "retriever_node"
    sm = get_streaming_manager()
    sm.emit_node_start(state.survey_id, "retriever_node", "Generating paper descriptions")
    sm.emit_log(state.survey_id, "retriever_node", "Retriever Node: Starting gap-aware paper description generation")

    try:
        if not state.tsv_path:
            sm.emit_log(state.survey_id, "retriever_node", "Retriever Node: No TSV path found, skipping")
            state.progress = 0.5
            state.current_node = "classifier_node"
            return state

        embedder = get_embedder()
        chroma = get_chroma_manager()
        llm = LLMClient()
        max_workers = DEFAULT_MAX_WORKERS

        gap_text = _build_gap_text(state)
        sentence_patterns = _generate_sentence_patterns(llm, state, gap_text)
        state.sentence_patterns = sentence_patterns
        sm.emit_log(state.survey_id, "retriever_node", f"Retriever Node: Generated {len(sentence_patterns)} sentence patterns")

        description_list, citation_data_list = _generate_descriptions_parallel(
            state=state,
            embedder=embedder,
            chroma=chroma,
            llm=llm,
            max_workers=max_workers,
            sentence_patterns=sentence_patterns,
            gap_text=gap_text,
        )

        state.description_list = description_list
        state.retrieval_results = description_list
        state.retrieval_results_enriched = _build_enriched_results(
            description_list, state.citation_paper_meta
        )
        state.citation_data_list = citation_data_list
        sm.emit_log(state.survey_id, "retriever_node", f"Retriever Node: Generated {len(description_list)} descriptions")

        citation_data_path = _save_citation_data(
            state=state,
            description_list=description_list,
            citation_data_list=citation_data_list,
        )
        state.citation_data_path = citation_data_path

        tsv_mgr = TSVManager()
        tsv_with_retrieval = tsv_mgr.update_retrieval_results(
            state.tsv_path,
            description_list,
        )
        state.tsv_with_retrieval = tsv_with_retrieval

        state.progress = 0.5
        state.current_node = "classifier_node"
        sm.emit_log(state.survey_id, "retriever_node", "Retriever Node: Complete, transitioning to Classifier Node")

    except Exception as e:
        sm.emit_log(state.survey_id, "retriever_node", f"Retriever Node: ERROR - {str(e)}")
        sm.emit_error(state.survey_id, "retriever_node", str(e))
        state.error_message = str(e)
        state.status = "FAILED"

    return state


def _build_gap_text(state: SurveyState) -> str:
    gap_value = state.gap_type.value if state.gap_type != GapType.EMPTY else ""
    gap_desc = GAP_TYPE_DESCRIPTIONS.get(state.gap_type, "") if gap_value else ""
    if gap_value and gap_desc:
        return f"Topic: {state.user_input}\nGap Type: {gap_value}\nGap Description: {gap_desc}"
    if gap_value:
        return f"Topic: {state.user_input}\nGap Type: {gap_value}"
    return state.user_input


def _generate_sentence_patterns(llm: LLMClient, state: SurveyState, gap_text: str) -> list[str]:
    sm = get_streaming_manager()
    prompt = SENTENCE_PATTERN_USER_TEMPLATE.format(
        gap_text=gap_text,
        num_patterns=5,
    )

    def on_chunk(token: str):
        sm.emit_llm_token(state.survey_id, "retriever_node", token)

    response = llm.generate(
        system_prompt=SENTENCE_PATTERN_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=512,
        temperature=0.7,
        model="glm-5.1",
        on_chunk=on_chunk,
    )
    try:
        patterns = json.loads(response)
        if isinstance(patterns, list):
            cleaned = [str(item).strip() for item in patterns if str(item).strip()]
            if cleaned:
                return cleaned[:5]
    except json.JSONDecodeError:
        pass
    fallback = [
        gap_text,
        f"This work addresses {gap_text} through",
        f"The proposed method for {gap_text} is based on",
    ]
    return fallback[:5]


def _generate_descriptions_parallel(
    state: SurveyState,
    embedder,
    chroma,
    llm: LLMClient,
    max_workers: int,
    sentence_patterns: list[str],
    gap_text: str,
) -> tuple[dict[str, str], dict[str, list[dict]]]:
    descriptions = {}
    citation_data = {}

    def gen_for_collection(collection_name: str) -> tuple[str, str, list[dict]]:
        try:
            citation_chunks = _retrieve_citation_chunks_for_collection(
                collection_name=collection_name,
                sentence_patterns=sentence_patterns,
                embedder=embedder,
                chroma=chroma,
            )
            description = _generate_gap_aware_description(
                llm=llm,
                state=state,
                title=collection_name,
                topic=state.user_input,
                gap_text=gap_text,
                citation_chunks=citation_chunks,
            )
            return collection_name, description, citation_chunks
        except Exception as e:
            state.add_log(f"Retriever Node: Failed description for {collection_name}: {e}")
            return collection_name, "", []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(gen_for_collection, coll): coll
            for coll in state.chroma_collections
        }
        for future in concurrent.futures.as_completed(futures):
            coll, description, citation_chunks = future.result()
            descriptions[coll] = description
            citation_data[coll] = citation_chunks

    return descriptions, citation_data


def _build_enriched_results(
    description_list: dict[str, str],
    citation_paper_meta: dict,
) -> dict[str, dict]:
    enriched = {}
    for coll_name, description in description_list.items():
        meta = citation_paper_meta.get(coll_name, {}) if citation_paper_meta else {}
        enriched[coll_name] = {
            "description": description,
            "full_title": meta.get("title", ""),
            "authors": meta.get("authors", []),
            "year": meta.get("year"),
        }
    return enriched


def _retrieve_citation_chunks_for_collection(
    collection_name: str,
    sentence_patterns: list[str],
    embedder,
    chroma,
) -> list[dict]:
    merged_chunks = []
    for pattern in sentence_patterns:
        q_emb = embedder.embed_query(pattern)
        result = chroma.query(
            collection_name=collection_name,
            query_embeddings=[q_emb],
            n_results=DEFAULT_N_RESULTS,
        )
        merged_chunks.extend(_extract_citation_entries(collection_name, pattern, result))
    return _dedupe_citation_chunks(merged_chunks)


def _extract_citation_entries(collection_name: str, query_pattern: str, result: dict) -> list[dict]:
    if not result or not result.get("documents"):
        return []

    documents = result.get("documents", [[]])
    metadatas = result.get("metadatas", [[]])
    distances = result.get("distances", [[]])
    ids = result.get("ids", [[]])

    docs = documents[0] if documents else []
    metas = metadatas[0] if metadatas else []
    dists = distances[0] if distances else []
    chunk_ids = ids[0] if ids else []

    entries = []
    for idx, content in enumerate(docs):
        metadata = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
        distance = dists[idx] if idx < len(dists) else None
        chunk_id = chunk_ids[idx] if idx < len(chunk_ids) else metadata.get("chunk_id", "")
        doc_name = metadata.get("doc_name", collection_name)
        chunk_index = metadata.get("chunk_index")
        if chunk_index is not None:
            source = f"{doc_name}#chunk_{chunk_index}"
        elif chunk_id:
            source = f"{doc_name}#{chunk_id}"
        else:
            source = doc_name
        entries.append(
            {
                "source": source,
                "distance": float(distance) if distance is not None else None,
                "content": content,
                "collection_name": collection_name,
                "query_pattern": query_pattern,
            }
        )
    return entries


def _dedupe_citation_chunks(chunks: list[dict]) -> list[dict]:
    unique = {}
    for item in chunks:
        content = (item.get("content") or "").strip()
        source = item.get("source") or ""
        if not content:
            continue
        key = (content, source)
        current = unique.get(key)
        current_distance = current.get("distance") if current else None
        item_distance = item.get("distance")
        if current is None:
            unique[key] = item
            continue
        if current_distance is None:
            continue
        if item_distance is not None and item_distance < current_distance:
            unique[key] = item

    sorted_items = sorted(
        unique.values(),
        key=lambda x: x.get("distance") if x.get("distance") is not None else float("inf"),
    )
    return sorted_items[:8]


def _generate_gap_aware_description(
    llm: LLMClient,
    state: SurveyState,
    title: str,
    topic: str,
    gap_text: str,
    citation_chunks: list[dict],
) -> str:
    evidence_lines = []
    total_chars = 0
    max_chars = 3000
    for idx, chunk in enumerate(citation_chunks, start=1):
        snippet = (chunk.get("content") or "").strip()
        if not snippet:
            continue
        line = f"[{idx}] Source: {chunk.get('source', title)}\n{snippet[:500]}"
        if total_chars + len(line) > max_chars:
            break
        evidence_lines.append(line)
        total_chars += len(line)

    evidence_text = "\n\n".join(evidence_lines) if evidence_lines else "No relevant evidence retrieved."
    prompt = GAP_PAPER_DESCRIPTION_USER_TEMPLATE.format(
        title=title,
        topic=topic,
        gap_text=gap_text,
        evidence=evidence_text,
    )
    sm = get_streaming_manager()

    def on_chunk(token: str):
        sm.emit_llm_token(state.survey_id, "retriever_node", token)

    return llm.generate(
        system_prompt=GAP_PAPER_DESCRIPTION_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=256,
        temperature=0.3,
        model="deepseek-v4-pro",
        on_chunk=on_chunk,
    ).strip()


def _save_citation_data(
    state: SurveyState,
    description_list: dict[str, str],
    citation_data_list: dict[str, list[dict]],
) -> str:
    info_dir = Path("./data/info") / state.survey_id
    info_dir.mkdir(parents=True, exist_ok=True)
    citation_path = info_dir / "citation_data.json"

    payload = {
        "survey_id": state.survey_id,
        "topic": state.user_input,
        "gap_type": state.gap_type.value,
        "sentence_patterns": state.sentence_patterns,
        "papers": {
            paper_id: {
                "description": description_list.get(paper_id, ""),
                "citations": citation_data_list.get(paper_id, []),
            }
            for paper_id in description_list
        },
    }
    citation_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    state.add_log(f"Retriever Node: Saved citation data to {citation_path}")
    return str(citation_path)

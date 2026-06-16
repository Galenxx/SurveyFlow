"""
Gap Node: Analyze research gaps using vector retrieval + LLM analysis.
"""
from core.state import SurveyState, GapType
from core.llm_client import LLMClient
from core.embedder import get_embedder
from core.chroma_client import get_chroma_manager
from core.constants import GAP_TYPE_DESCRIPTIONS, GAP_ANALYSIS_QUERIES, DEFAULT_N_RESULTS
from streaming.manager import get_streaming_manager
from prompts.gap_prompts import GAP_SYSTEM_PROMPT_TEMPLATE, GAP_USER_PROMPT_TEMPLATE


def gap_node(state: SurveyState) -> SurveyState:
    """Perform gap analysis on retrieved chunks."""
    state.current_node = "gap_node"
    sm = get_streaming_manager()
    sm.emit_node_start(state.survey_id, "gap_node", "Analyzing research gaps")
    sm.emit_log(state.survey_id, "gap_node", f"Gap Node: Starting gap analysis for type: {state.gap_type.value}")

    try:
        if state.gap_type == GapType.EMPTY:
            state.gap_type = GapType.METHODOLOGICAL
            sm.emit_log(state.survey_id, "gap_node", "Gap Node: No gap specified, defaulting to Methodological Gap")

        gap_queries = _build_gap_queries(state.gap_type, state.user_input)
        all_chunks = _retrieve_chunks(state, gap_queries)

        if not all_chunks:
            sm.emit_log(state.survey_id, "gap_node", "Gap Node: No chunks retrieved, generating report without evidence")
            all_chunks = ["No relevant chunks found for gap analysis."]

        state.gap_chunks = all_chunks
        gap_report = _generate_gap_report(state, all_chunks)
        state.gap_report = gap_report

        sm.emit_log(state.survey_id, "gap_node", "Gap Node: Gap analysis complete")
        state.progress = 0.4
        state.current_node = "retriever_node"

    except Exception as e:
        sm.emit_log(state.survey_id, "gap_node", f"Gap Node: ERROR - {str(e)}")
        sm.emit_error(state.survey_id, "gap_node", str(e))
        state.error_message = str(e)
        state.status = "FAILED"

    return state


def _build_gap_queries(gap_type: GapType, user_input: str) -> list[str]:
    base_queries = GAP_ANALYSIS_QUERIES.get(gap_type, GAP_ANALYSIS_QUERIES[GapType.METHODOLOGICAL])
    return base_queries


def _retrieve_chunks(state: SurveyState, queries: list[str]) -> list[str]:
    embedder = get_embedder()
    chroma = get_chroma_manager()

    all_chunks = []
    seen = set()

    for query_text in queries:
        try:
            q_emb = embedder.embed_query(query_text)
            for coll_name in state.chroma_collections:
                try:
                    result = chroma.query(
                        collection_name=coll_name,
                        query_embeddings=[q_emb],
                        n_results=3,
                    )
                    if result and result.get("documents"):
                        for chunk in result["documents"][0]:
                            if chunk not in seen:
                                all_chunks.append(chunk)
                                seen.add(chunk)
                except Exception as e:
                    state.add_log(f"Gap Node: Chunk retrieval error for {coll_name} - {e}")
                    continue
        except Exception as e:
            state.add_log(f"Gap Node: Embedding error for query '{query_text[:30]}...' - {e}")
            continue

    return all_chunks[:50]


def _generate_gap_report(state: SurveyState, chunks: list[str]) -> str:
    llm = LLMClient()
    sm = get_streaming_manager()
    gap_type_value = state.gap_type.value
    gap_desc = GAP_TYPE_DESCRIPTIONS.get(state.gap_type, "")

    chunk_texts = "\n".join([
        f"[{i+1}] {chunk[:300]}" for i, chunk in enumerate(chunks[:30])
    ])

    system_prompt = GAP_SYSTEM_PROMPT_TEMPLATE.format(
        user_input=state.user_input,
        gap_type=gap_type_value,
        gap_description=f"\n{gap_desc}",
    )
    user_prompt = GAP_USER_PROMPT_TEMPLATE.format(chunks=chunk_texts)

    def on_chunk(token: str):
        sm.emit_llm_token(state.survey_id, "gap_node", token)

    report = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=8192,
        temperature=0.3,
        model="deepseek-v4-pro",
        on_chunk=on_chunk,
    )
    return report

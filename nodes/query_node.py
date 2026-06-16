"""
Query Node: Generates arXiv and Semantic Scholar search queries from natural language topic.
Based on asg_query.py patterns.
"""
import time
from core.state import SurveyState
from core.llm_client import LLMClient
from core.constants import DEFAULT_YEAR_START
from streaming.manager import get_streaming_manager
from prompts.query_prompts import (
    ABSTRACT_SYSTEM_PROMPT, ABSTRACT_USER_PROMPT_TEMPLATE,
    ENTITY_LIST_SYSTEM_PROMPT, ENTITY_LIST_USER_PROMPT_TEMPLATE,
    QUERY_SYSTEM_PROMPT, QUERY_USER_PROMPT_TEMPLATE,
    GENERIC_QUERY_SYSTEM_PROMPT, GENERIC_QUERY_USER_PROMPT_TEMPLATE,
    S2_QUERY_SYSTEM_PROMPT, S2_QUERY_USER_PROMPT_TEMPLATE,
    S2_GENERIC_QUERY_SYSTEM_PROMPT, S2_GENERIC_QUERY_USER_PROMPT_TEMPLATE,
)


def query_node(state: SurveyState) -> SurveyState:
    """Generate arXiv and Semantic Scholar queries from natural language topic."""
    state.current_node = "query_node"
    sm = get_streaming_manager()
    sm.emit_node_start(state.survey_id, "query_node", "Generating search queries")
    sm.emit_log(state.survey_id, "query_node", f"Query Node: Starting query generation for topic: {state.user_input}")

    try:
        llm = LLMClient()

        year_start = state.year_range[0] if state.year_range else DEFAULT_YEAR_START
        sm.emit_log(state.survey_id, "query_node", f"Query Node: Using year range from {year_start}")

        abstract_text = _generate_abstract(llm, state, topic=state.user_input)
        entity_list = _generate_entity_lists(llm, state, topic=state.user_input, abstract_text=abstract_text)

        arxiv_query = _generate_query(llm, state, topic=state.user_input, entity_list=entity_list)
        generic_arxiv_query = _generate_generic_query(llm, state, original_query=arxiv_query, topic=state.user_input)

        s2_query = _generate_s2_query(llm, state, topic=state.user_input, entity_list=entity_list)
        s2_generic_query = _generate_s2_generic_query(llm, state, original_query=s2_query, topic=state.user_input)

        state.arxiv_query = arxiv_query
        state.generic_arxiv_query = generic_arxiv_query
        state.s2_query = s2_query
        state.s2_generic_query = s2_generic_query
        state.query_building_log = (
            f"Topic: {state.user_input}, "
            f"Entities and Concepts: {entity_list}, "
            f"Year range: {year_start}+"
        )
        sm.emit_log(state.survey_id, "query_node", f"Query Node: Generated arXiv strict query - {arxiv_query[:100]}...")
        sm.emit_log(state.survey_id, "query_node", f"Query Node: Generated arXiv generic query - {generic_arxiv_query[:100]}...")
        sm.emit_log(state.survey_id, "query_node", f"Query Node: Generated S2 strict query - {s2_query[:100]}...")
        sm.emit_log(state.survey_id, "query_node", f"Query Node: Generated S2 generic query - {s2_generic_query[:100]}...")
        state.progress = 0.05
        state.current_node = "download_node"
    except Exception as e:
        sm.emit_log(state.survey_id, "query_node", f"Query Node: ERROR - {str(e)}")
        sm.emit_error(state.survey_id, "query_node", str(e))
        state.error_message = str(e)
        state.status = "FAILED"

    return state


def _generate_abstract(llm: LLMClient, state: SurveyState, topic: str, abstract_text: str = "") -> str:
    sm = get_streaming_manager()
    def on_chunk(token: str):
        sm.emit_llm_token(state.survey_id, "query_node", token)
    response = llm.generate(
        system_prompt=ABSTRACT_SYSTEM_PROMPT,
        user_prompt=ABSTRACT_USER_PROMPT_TEMPLATE.format(topic=topic),
        max_tokens=2048,
        temperature=0.5,
        model="deepseek-v4-pro",
        on_chunk=on_chunk,
    )
    return response


def _generate_entity_lists(llm: LLMClient, state: SurveyState, topic: str, abstract_text: str) -> str:
    sm = get_streaming_manager()
    def on_chunk(token: str):
        sm.emit_llm_token(state.survey_id, "query_node", token)
    response = llm.generate(
        system_prompt=ENTITY_LIST_SYSTEM_PROMPT,
        user_prompt=ENTITY_LIST_USER_PROMPT_TEMPLATE.format(
            topic=topic, abstract_text=abstract_text
        ),
        max_tokens=2048,
        temperature=0.5,
        model="deepseek-v4-pro",
        on_chunk=on_chunk,
    )
    return response


def _extract_query(response: str) -> str:
    import re

    text = response.strip()

    match = re.search(r"\(.*\)", text, re.DOTALL)
    if match:
        return match.group(0).strip()

    if not match:
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("Query:") or line.startswith("(abs:"):
                return line.lstrip("Query:").strip()

    return text.strip()


def _generate_query(llm: LLMClient, state: SurveyState, topic: str, entity_list: str) -> str:
    sm = get_streaming_manager()
    def on_chunk(token: str):
        sm.emit_llm_token(state.survey_id, "query_node", token)
    response = _generate_with_non_empty_retry(
        llm=llm,
        state=state,
        system_prompt=QUERY_SYSTEM_PROMPT,
        user_prompt=QUERY_USER_PROMPT_TEMPLATE.format(
            topic=topic, entity_list=entity_list
        ),
        extractor=_extract_query,
        max_tokens=2048,
        temperature=0.5,
        model="deepseek-v4-pro",
        on_chunk=on_chunk,
    )
    query = _extract_query(response)

    if not query:
        query = _fallback_arxiv_query(topic)

    return query


def _fallback_arxiv_query(topic: str) -> str:
    words = [w.strip().lower() for w in topic.split() if w.strip()]
    words = [w for w in words if len(w) > 2]
    selected = []
    seen = set()
    for word in words:
        if word not in seen:
            selected.append(word)
            seen.add(word)
        if len(selected) >= 4:
            break
    if not selected:
        selected = ["survey", "research"]
    return " AND ".join([f'abs:{word}' for word in selected])


def _generate_generic_query(llm: LLMClient, state: SurveyState, original_query: str, topic: str) -> str:
    sm = get_streaming_manager()
    def on_chunk(token: str):
        sm.emit_llm_token(state.survey_id, "query_node", token)
    response = _generate_with_non_empty_retry(
        llm=llm,
        state=state,
        system_prompt=GENERIC_QUERY_SYSTEM_PROMPT,
        user_prompt=GENERIC_QUERY_USER_PROMPT_TEMPLATE.format(
            original_query=original_query, topic=topic
        ),
        extractor=_extract_query,
        max_tokens=1024,
        temperature=0.5,
        model="deepseek-v4-pro",
        on_chunk=on_chunk,
    )
    query = _extract_query(response)

    if not query:
        query = _fallback_generic_arxiv_query(topic)

    return query


def _fallback_generic_arxiv_query(topic: str) -> str:
    words = [w.strip().lower() for w in topic.split() if w.strip()]
    words = [w for w in words if len(w) > 2]
    unique = []
    seen = set()
    for word in words:
        if word not in seen:
            unique.append(word)
            seen.add(word)
        if len(unique) >= 4:
            break
    while len(unique) < 4:
        unique.append("survey")
    return f'(abs:"{unique[0]}" AND abs:"{unique[1]}") OR (abs:"{unique[2]}" AND abs:"{unique[3]}")'


def _extract_s2_query(response: str) -> str:
    import re

    text = response.strip()

    match = re.search(r"Query:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"S2\s*Query:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            return line

    return text.strip()


def _generate_s2_query(llm: LLMClient, state: SurveyState, topic: str, entity_list: str) -> str:
    sm = get_streaming_manager()
    def on_chunk(token: str):
        sm.emit_llm_token(state.survey_id, "query_node", token)
    response = _generate_with_non_empty_retry(
        llm=llm,
        state=state,
        system_prompt=S2_QUERY_SYSTEM_PROMPT,
        user_prompt=S2_QUERY_USER_PROMPT_TEMPLATE.format(
            topic=topic, entity_list=entity_list
        ),
        extractor=_extract_s2_query,
        max_tokens=1024,
        temperature=0.5,
        model="deepseek-v4-pro",
        on_chunk=on_chunk,
    )
    query = _extract_s2_query(response)

    if not query:
        query = _fallback_s2_query(topic)

    return query


def _generate_s2_generic_query(llm: LLMClient, state: SurveyState, original_query: str, topic: str) -> str:
    sm = get_streaming_manager()
    def on_chunk(token: str):
        sm.emit_llm_token(state.survey_id, "query_node", token)
    response = _generate_with_non_empty_retry(
        llm=llm,
        state=state,
        system_prompt=S2_GENERIC_QUERY_SYSTEM_PROMPT,
        user_prompt=S2_GENERIC_QUERY_USER_PROMPT_TEMPLATE.format(
            original_query=original_query, topic=topic
        ),
        extractor=_extract_s2_query,
        max_tokens=1024,
        temperature=0.5,
        model="deepseek-v4-pro",
        on_chunk=on_chunk,
    )
    query = _extract_s2_query(response)

    if not query:
        query = _fallback_s2_query(topic)

    return query


def _fallback_s2_query(topic: str) -> str:
    words = [w.strip().lower() for w in topic.split() if w.strip()]
    words = [w for w in words if len(w) > 2]
    unique = []
    seen = set()
    for word in words:
        if word not in seen:
            unique.append(word)
            seen.add(word)
        if len(unique) >= 6:
            break
    if not unique:
        unique = ["survey", "research", "method"]
    return " ".join(unique)


def _generate_with_non_empty_retry(
    llm: LLMClient,
    state: SurveyState,
    system_prompt: str,
    user_prompt: str,
    extractor,
    max_tokens: int,
    temperature: float,
    model: str,
    retries: int = 3,
    on_chunk: callable = None,
) -> str:
    sm = get_streaming_manager()
    last_response = ""
    for attempt in range(retries):
        def make_callback(suffix: str):
            def cb(token: str):
                sm.emit_llm_token(state.survey_id, "query_node", token)
                if on_chunk:
                    on_chunk(token)
            return cb

        response = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            on_chunk=make_callback(""),
        )
        last_response = response
        extracted = extractor(response)
        if extracted:
            return response
        if attempt < retries - 1:
            time.sleep(1.0 * (attempt + 1))
    return last_response

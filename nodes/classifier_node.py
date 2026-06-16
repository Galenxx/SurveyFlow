"""
Classifier Node: Classify papers into categories using LLM.
Based on asg_clustername.py patterns.
"""
import json
import re
from core.state import SurveyState
from core.llm_client import LLMClient
from core.tsv_manager import TSVManager
from streaming.manager import get_streaming_manager
from prompts.classification_prompts import CLASSIFICATION_SYSTEM_PROMPT, CLASSIFICATION_USER_PROMPT_TEMPLATE


def classifier_node(state: SurveyState) -> SurveyState:
    """Classify papers based on their descriptions."""
    state.current_node = "classifier_node"
    sm = get_streaming_manager()
    sm.emit_node_start(state.survey_id, "classifier_node", "Classifying papers")
    sm.emit_log(state.survey_id, "classifier_node", "Classifier Node: Starting paper classification")

    try:
        tsv_path = state.tsv_with_retrieval or state.tsv_path
        if not tsv_path:
            sm.emit_log(state.survey_id, "classifier_node", "Classifier Node: No TSV path found, using default 3 clusters")
            state.num_classes = 3
            state.labels = {}
            state.progress = 0.55
            state.current_node = "section_node"
            return state

        llm = LLMClient()
        paper_list = _build_paper_list(state)

        if not paper_list:
            sm.emit_log(state.survey_id, "classifier_node", "Classifier Node: No papers to classify")
            state.num_classes = 3
            state.labels = {}
            state.progress = 0.55
            state.current_node = "section_node"
            return state

        classification = _classify_papers(llm, state, paper_list)

        state.num_classes = classification.get("num_classes", 3)
        state.labels = classification.get("paper_classifications", {})
        state.class_definitions = classification.get("class_definitions", {})

        sm.emit_log(state.survey_id, "classifier_node", f"Classifier Node: Classified into {state.num_classes} categories")

        tsv_mgr = TSVManager()
        tsv_with_labels = tsv_mgr.update_labels(tsv_path, state.labels)
        state.tsv_with_labels = tsv_with_labels

        state.progress = 0.55
        state.current_node = "section_node"

    except Exception as e:
        sm.emit_log(state.survey_id, "classifier_node", f"Classifier Node: ERROR - {str(e)}")
        sm.emit_error(state.survey_id, "classifier_node", str(e))
        state.error_message = str(e)
        state.status = "FAILED"

    return state


def _build_paper_list(state: SurveyState) -> str:
    papers = []
    for coll in state.chroma_collections:
        retrieval = state.retrieval_results.get(coll, "")
        if not retrieval:
            retrieval = state.user_input
        papers.append(f"[{len(papers)}] {coll}: {retrieval}")
    return "\n".join(papers[:30])


def _classify_papers(llm: LLMClient, state: SurveyState, paper_list: str) -> dict:
    """Call LLM to classify papers."""
    sm = get_streaming_manager()
    prompt = CLASSIFICATION_USER_PROMPT_TEMPLATE.format(
        survey_title=state.user_input,
        paper_list=paper_list,
    )

    def on_chunk(token: str):
        sm.emit_llm_token(state.survey_id, "classifier_node", token)

    response = llm.generate(
        system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=8192,
        temperature=0.3,
        model="glm-5.1",
        on_chunk=on_chunk,
    )

    parsed = _parse_classification_response(response)

    paper_classifications = parsed.get("paper_classifications", {})
    if not paper_classifications:
        raise ValueError(f"Classifier LLM returned empty paper_classifications. Response: {response[:500]}")

    labels: dict[str, int] = {}
    for class_id_str, papers in paper_classifications.items():
        class_id = int(class_id_str)
        for title in papers:
            labels[title] = class_id

    return {
        "num_classes": parsed.get("num_classes", 3),
        "class_definitions": parsed.get("class_definitions", {}),
        "paper_classifications": labels,
    }


def _parse_classification_response(response: str) -> dict:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    text = response.strip()
    for end in range(len(text), 0, -1):
        candidate = text[:end]
        try:
            result = json.loads(candidate)
            if "num_classes" in result or "paper_classifications" in result:
                return result
        except json.JSONDecodeError:
            pass

    fixed = re.sub(r",(\s*[}\]])", r"\1", text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    if open_braces > 0:
        fixed = text + ("}" * open_braces)
    elif open_brackets > 0:
        fixed = text + ("]" * open_brackets)
    else:
        fixed = text
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Classifier LLM did not return parseable JSON: {response[:500]}")

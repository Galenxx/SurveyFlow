"""
Section Node: Generate section titles for each paper category.
Based on asg_clustername.py patterns.
"""
import json
import re
from core.state import SurveyState
from core.llm_client import LLMClient
from streaming.manager import get_streaming_manager
from prompts.section_prompts import (
    SECTION_SYSTEM_PROMPT_TEMPLATE,
    SECTION_USER_PROMPT_TEMPLATE,
    SECTION_REFINEMENT_PROMPT,
)


def section_node(state: SurveyState) -> SurveyState:
    """Generate section titles based on paper classifications."""
    state.current_node = "section_node"
    sm = get_streaming_manager()
    sm.emit_node_start(state.survey_id, "section_node", "Generating section titles")
    sm.emit_log(state.survey_id, "section_node", "Section Node: Starting section title generation")

    try:
        llm = LLMClient()
        num_classes = state.num_classes or 3
        cluster_info = _build_cluster_info(state, num_classes)

        state.cluster_info = cluster_info

        cluster_info_text = _format_cluster_info(cluster_info)
        prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            survey_title=state.user_input,
            cluster_info=cluster_info_text,
        )
        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(survey_title=state.user_input)

        def on_chunk(token: str):
            sm.emit_llm_token(state.survey_id, "section_node", token)

        response = llm.generate(
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_tokens=512,
            temperature=0.3,
            model="deepseek-v4-pro",
            on_chunk=on_chunk,
        )

        titles = _parse_titles(response)
        if len(titles) < num_classes:
            titles = titles + [f"Section {i+1}" for i in range(len(titles), num_classes)]

        state.section_titles = titles[:num_classes]
        for i, title in enumerate(state.section_titles):
            if i in cluster_info:
                cluster_info[i]["name"] = title

        sm.emit_log(state.survey_id, "section_node", f"Section Node: Generated titles - {state.section_titles}")

        refinement_prompt = SECTION_REFINEMENT_PROMPT.format(
            section_titles=state.section_titles
        )
        refined = llm.generate(
            system_prompt="You are an academic paper writing expert.",
            user_prompt=refinement_prompt,
            max_tokens=512,
            temperature=0.3,
            model="deepseek-v4-pro",
            on_chunk=on_chunk,
        )
        refined_titles = _parse_titles(refined)
        if len(refined_titles) == num_classes:
            state.section_titles = refined_titles
            for i, title in enumerate(state.section_titles):
                if i in cluster_info:
                    cluster_info[i]["name"] = title

        state.progress = 0.6
        state.current_node = "outline_node"

    except Exception as e:
        sm.emit_log(state.survey_id, "section_node", f"Section Node: ERROR - {str(e)}")
        sm.emit_error(state.survey_id, "section_node", str(e))
        state.error_message = str(e)
        state.status = "FAILED"

    return state


def _build_cluster_info(state: SurveyState, num_classes: int) -> dict:
    enriched = getattr(state, "retrieval_results_enriched", None) or {}
    info = {}
    for paper_id, class_id in state.labels.items():
        try:
            label = int(class_id)
        except (ValueError, TypeError):
            continue
        if label not in info:
            info[label] = {"name": "", "titles": [], "descriptions": []}
        paper_id_str = str(paper_id)
        if enriched and paper_id_str in enriched:
            entry = enriched[paper_id_str]
            full_title = entry.get("full_title", paper_id_str)
            info[label]["titles"].append(full_title)
            info[label]["descriptions"].append(entry.get("description", ""))
        else:
            desc = state.retrieval_results.get(paper_id_str, "")
            info[label]["titles"].append(paper_id_str)
            info[label]["descriptions"].append(desc)
    for i in range(num_classes):
        if i not in info:
            info[i] = {"name": "", "titles": [], "descriptions": []}
    return info


def _format_cluster_info(cluster_info: dict) -> str:
    lines = []
    for label, data in sorted(cluster_info.items()):
        titles = data.get("titles", [])
        descs = data.get("descriptions", [])
        parts = []
        for i, (t, d) in enumerate(zip(titles, descs[:3])):
            parts.append(f"  Paper: {t}\n  Description: {d[:150]}")
        desc_text = "\n".join(parts) if parts else "(no papers)"
        lines.append(f"Category {label}:\n{desc_text}")
    return "\n\n".join(lines)


def _parse_titles(response: str) -> list[str]:
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return []

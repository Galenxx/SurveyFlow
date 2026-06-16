"""
Outline Node: Generate hierarchical survey outline.
Based on asg_outline.py patterns.
"""
import ast
import re
from core.state import SurveyState
from core.llm_client import LLMClient
from streaming.manager import get_streaming_manager
from prompts.outline_prompts import OUTLINE_SYSTEM_PROMPT, OUTLINE_USER_PROMPT_TEMPLATE


def outline_node(state: SurveyState) -> SurveyState:
    """Generate survey outline based on section titles and cluster info."""
    state.current_node = "outline_node"
    sm = get_streaming_manager()
    sm.emit_node_start(state.survey_id, "outline_node", "Generating survey outline")
    sm.emit_log(state.survey_id, "outline_node", "Outline Node: Starting outline generation")

    try:
        llm = LLMClient()
        num_classes = state.num_classes or 3
        cluster_with_claims = _build_claims(state)
        cluster_budget = _build_cluster_budget(state, num_classes)

        cluster_sections = _build_cluster_sections(state.section_titles, num_classes)
        first_level = f"[[1, '1 Abstract'], [1, '2 Introduction'], {cluster_sections}, [1, '{num_classes+3} Future Directions'], [1, '{num_classes+4} Conclusion'], [1, '{num_classes+5} References']]"

        prompt = OUTLINE_USER_PROMPT_TEMPLATE.format(
            survey_title=state.user_input,
            cluster_with_claims=cluster_with_claims,
            cluster_budget=cluster_budget,
            first_level_sections=first_level,
        )

        def on_chunk(token: str):
            sm.emit_llm_token(state.survey_id, "outline_node", token)

        response = llm.generate(
            system_prompt=OUTLINE_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=8192,
            temperature=0.3,
            model="glm-5.1",
            on_chunk=on_chunk,
        )

        state.outline_raw = response
        outline = _parse_outline(response)
        if outline:
            outline = _filter_macro_subsections(outline, num_classes)
            outline = _enforce_two_level_structure(outline, num_classes)
            state.outline = outline
            sm.emit_log(state.survey_id, "outline_node", f"Outline Node: Generated outline with {len(outline)} sections")
        else:
            sm.emit_log(state.survey_id, "outline_node", "Outline Node: Failed to parse outline, using default")

        state.progress = 0.65
        state.current_node = "writer_node"

    except Exception as e:
        sm.emit_log(state.survey_id, "outline_node", f"Outline Node: ERROR - {str(e)}")
        sm.emit_error(state.survey_id, "outline_node", str(e))
        state.error_message = str(e)
        state.status = "FAILED"

    return state


def _build_claims(state: SurveyState) -> str:
    lines = []
    for label, info in state.cluster_info.items():
        name = state.section_titles[label] if label < len(state.section_titles) else f"Cluster {label}"
        claims = "\n".join(f"- {d}" for d in info.get("descriptions", [])[:10])
        lines.append(f"## Cluster {label}: {name}\n{claims}\n")
    return "\n".join(lines)


def _l2_budget_for_cluster(num_papers: int) -> int:
    """Compute how many L2 subsections a cluster should have.

    Rule: 1 L2 per paper, capped at 3, with a floor of 1. This keeps each
    subsection substantive (roughly 1-3 papers per L2) and avoids the
    over-granularity of the previous 2-L2 x 3-L3 = 6 leaves per cluster.
    """
    if num_papers <= 0:
        return 1
    return min(3, max(1, num_papers))


def _build_cluster_budget(state: SurveyState, num_classes: int) -> str:
    """Format the per-cluster L2 budget block for inclusion in the user prompt."""
    lines = []
    for label in range(num_classes):
        info = state.cluster_info.get(label, {})
        titles = info.get("titles", []) if isinstance(info, dict) else []
        budget = _l2_budget_for_cluster(len(titles))
        lines.append(f"- Cluster {label} (section {label + 3}): {budget} level-2 subsection(s)")
    return "\n".join(lines)


def _build_cluster_sections(section_titles: list[str], num_classes: int) -> str:
    parts = []
    for i, title in enumerate(section_titles[:num_classes]):
        parts.append(f"[1, '{i+3} {title}']")
    return ", ".join(parts)


def _parse_outline(response: str) -> list:
    match = re.search(r"\[(.*)\]", response, re.DOTALL)
    if match:
        try:
            parsed = ast.literal_eval("[" + match.group(1) + "]")
            if parsed and all(isinstance(item, list) and len(item) == 2 for item in parsed):
                return [(int(item[0]), str(item[1])) for item in parsed]
        except (ValueError, SyntaxError):
            pass
    return []


def _filter_macro_subsections(outline: list, num_classes: int) -> list:
    """Drop level-2 entries that belong to macro (non-cluster) sections.

    Cluster sections are numbered ``3 .. num_classes+2``; everything else
    (1=Abstract, 2=Introduction, and the trailing Future Directions /
    Conclusion / References at ``num_classes+3 ..``) is a macro section
    whose subsections we discard. The old code hard-coded the macro set
    as ``{1, 2, 6, 7, 8}``, which assumed exactly 3 clusters: when the
    classifier produced 2 or 4 clusters the cluster/macro boundary shifted
    and real cluster subsections were silently deleted (or macro junk
    survived), starving the writer of sections to ground citations in.
    Deriving the boundary from ``num_classes`` keeps the filter correct
    for any cluster count.
    """
    cluster_lo = 3
    cluster_hi = num_classes + 2  # inclusive
    filtered = []
    for level, title in outline:
        if level == 1:
            filtered.append((level, title))
            continue
        m = re.match(r"^(\d+)", title.strip())
        first_digit = int(m.group(1)) if m else None
        is_cluster = first_digit is not None and cluster_lo <= first_digit <= cluster_hi
        if not is_cluster:
            # macro subsection (or unparseable number) -> drop
            continue
        filtered.append((level, title))
    return filtered


def _enforce_two_level_structure(outline: list, num_classes: int) -> list:
    """Drop any level-3 entries the LLM may have emitted despite the prompt.

    The new outline policy is two levels deep: L1 (Abstract/Introduction/
    cluster/Conclusion/etc.) and L2 (cluster subsections). Any level >= 3
    entries that survive the macro filter are removed here as a safety net.
    """
    return [(level, title) for level, title in outline if level <= 2]

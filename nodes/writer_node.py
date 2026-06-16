"""
Writer Node: Generate full survey paper content including all sections.
Based on survey_generator_api.py and asg_outline.py patterns.

Generation order:
  Step 1: Generate section content for level >= 2 outlines in parallel (cluster-scoped).
  Step 2: Generate Introduction based on section content.
  Step 3: Generate Abstract based on section content.
  Step 4: Generate Conclusion based on section content.
  Step 5: Generate Future Directions based on section content.
  Step 6: Generate References from used papers.
  Step 7: Assemble and save md file.
  Step 7b-7f: Re-generate macro sections from assembled survey text (ensures consistency).
  Step 7g: Final assembly and save.

Citation handling:
  - Section body prompts (SECTION_WRITING_USER_TEMPLATE): context chunks are
    prepended with APA-format source labels, and the LLM is instructed to embed
    in-text citations (Frej et al., 2024) throughout the prose.
  - Abstract, Introduction, Conclusion, Future Directions: these are generated
    twice — first from raw section content for macro generation, then again
    from the fully assembled survey text to ensure full consistency. No
    ChromaDB retrieval is used for these sections.
  - References section: generated from paper metadata in APA 7th edition format.
"""
import concurrent.futures
import re
from pathlib import Path
from core.state import SurveyState
from core.llm_client import LLMClient
from core.embedder import get_embedder
from core.chroma_client import get_chroma_manager, _derive_collection_name
from core.constants import DEFAULT_MAX_WORKERS
from streaming.manager import get_streaming_manager
from prompts.writer_prompts import (
    SECTION_WRITING_SYSTEM_PROMPT,
    SECTION_WRITING_USER_TEMPLATE,
    INTRODUCTION_USER_TEMPLATE,
    ABSTRACT_USER_TEMPLATE,
    CONCLUSION_USER_TEMPLATE,
    FUTURE_WORK_USER_TEMPLATE,
    REFERENCE_USER_TEMPLATE,
)


def writer_node(state: SurveyState) -> SurveyState:
    """Generate all survey sections and assemble the complete paper."""
    state.current_node = "writer_node"
    sm = get_streaming_manager()
    sm.emit_node_start(state.survey_id, "writer_node", "Writing survey paper")
    sm.emit_log(state.survey_id, "writer_node", "Writer Node: Starting survey generation")

    try:
        llm = LLMClient()
        embedder = get_embedder()
        chroma = get_chroma_manager()

        if state.enable_citations and state.citation_paper_meta:
            _paper_meta_map = state.citation_paper_meta.copy()
            _enrich_from_citation_data(state, _paper_meta_map)
            sm.emit_log(state.survey_id, "writer_node",
                f"Writer Node: Using citation_paper_meta for {len(_paper_meta_map)} papers")
        else:
            _paper_meta_map = _derive_paper_map(state, chroma)
            _enrich_from_citation_data(state, _paper_meta_map)

        sm.emit_log(state.survey_id, "writer_node", "Writer Node: [Step 1] Generating section content (level >= 2) in parallel")
        section_content = _generate_sections(state, llm, embedder, chroma, _paper_meta_map)

        sm.emit_log(state.survey_id, "writer_node", "Writer Node: [Step 2] Generating introduction")
        state.introduction = _generate_introduction(state, llm, section_content)

        sm.emit_log(state.survey_id, "writer_node", "Writer Node: [Step 3] Generating abstract (based on section content)")
        state.abstract = _generate_abstract(state, llm, section_content)

        sm.emit_log(state.survey_id, "writer_node", "Writer Node: [Step 4] Generating conclusion")
        state.conclusion = _generate_conclusion(state, llm, section_content)

        sm.emit_log(state.survey_id, "writer_node", "Writer Node: [Step 5] Generating future directions")
        state.future_directions = _generate_future_directions(state, llm, section_content)

        sm.emit_log(state.survey_id, "writer_node", "Writer Node: [Step 6] Generating references")
        state.references = _generate_references(state, llm, _paper_meta_map)

        sm.emit_log(state.survey_id, "writer_node", "Writer Node: [Step 7] Assembling paper and saving md file")
        md_file_path, survey_text = _assemble_and_save(state, section_content, _paper_meta_map)
        state.survey_text = survey_text
        state.md_file_path = md_file_path

        sm.emit_log(state.survey_id, "writer_node", "Writer Node: [Step 7b] Re-generating macro sections from assembled survey")
        assembled_text = _read_assembled_survey(md_file_path)
        sm.emit_log(state.survey_id, "writer_node", "Writer Node: [Step 7c] Re-generating abstract")
        state.abstract = _generate_abstract_from_assembled(state, llm, assembled_text)
        sm.emit_log(state.survey_id, "writer_node", "Writer Node: [Step 7d] Re-generating introduction")
        state.introduction = _generate_introduction_from_assembled(state, llm, assembled_text)
        sm.emit_log(state.survey_id, "writer_node", "Writer Node: [Step 7e] Re-generating conclusion")
        state.conclusion = _generate_conclusion_from_assembled(state, llm, assembled_text)
        sm.emit_log(state.survey_id, "writer_node", "Writer Node: [Step 7f] Re-generating future directions")
        state.future_directions = _generate_future_from_assembled(state, llm, assembled_text)

        sm.emit_log(state.survey_id, "writer_node", "Writer Node: [Step 7g] Final assembly and save")
        md_file_path, survey_text = _assemble_and_save(state, section_content, _paper_meta_map)
        state.survey_text = survey_text
        state.md_file_path = md_file_path

        sm.emit_log(state.survey_id, "writer_node", "Writer Node: Survey generation complete")
        state.status = "COMPLETED"
        state.progress = 1.0

    except Exception as e:
        sm.emit_log(state.survey_id, "writer_node", f"Writer Node: ERROR - {str(e)}")
        sm.emit_error(state.survey_id, "writer_node", str(e))
        state.error_message = str(e)
        state.status = "FAILED"

    return state


# ------------------------------------------------------------------
# Paper metadata mapping
# ------------------------------------------------------------------

def _format_apa_short(meta: dict) -> str:
    authors: list = meta.get("authors", [])
    year = meta.get("year") or "n.d."
    year_str = str(year) if year != "n.d." else "n.d."

    if not authors:
        title = meta.get("title", "")
        truncated = title[:60].strip()
        return f"{truncated} ({year_str})"

    if len(authors) == 1:
        first = authors[0].split()[-1]
        return f"{first} ({year_str})"
    elif len(authors) == 2:
        a1 = authors[0].split()[-1]
        a2 = authors[1].split()[-1]
        return f"{a1} & {a2} ({year_str})"
    else:
        first = authors[0].split()[-1]
        return f"{first} et al. ({year_str})"


def _enrich_from_citation_data(state: SurveyState, paper_meta_map: dict) -> None:
    cdl = getattr(state, "citation_data_list", None) or {}
    for cn, entries in cdl.items():
        if not entries:
            continue

        existing = paper_meta_map.get(cn)
        if existing and existing.get("authors") and existing.get("year"):
            continue

        first_entry = entries[0]
        source = first_entry.get("source", "")
        title_from_source = source.split("#")[0].strip() if source else ""

        year_found = None
        chunk_content = first_entry.get("content", "")[:1000]
        year_matches = re.findall(r"\((\d{4})\)|\b(\d{4})\b|\[(\d{4})\]", chunk_content)
        flat = [g for m in year_matches for g in m if g]
        if flat:
            valid_years = [int(y) for y in flat if 2015 <= int(y) <= 2026]
            if valid_years:
                year_found = max(valid_years)

        if cn not in paper_meta_map:
            if title_from_source:
                paper_meta_map[cn] = {
                    "title": title_from_source,
                    "authors": [],
                    "year": year_found,
                    "arxiv_id": "",
                    "s2_id": "",
                }
        else:
            entry = paper_meta_map[cn]
            if not entry.get("authors") and not entry.get("year"):
                if title_from_source:
                    entry["title"] = title_from_source
            if not entry.get("year") and year_found:
                entry["year"] = year_found


def _derive_paper_map(state: SurveyState, chroma) -> dict:
    paper_map = {}
    for paper in state.paper_list:
        title = paper.get("title", "")
        if not title:
            continue
        cn = _derive_collection_name(title)
        if cn in state.chroma_collections:
            try:
                chroma.client.get_collection(name=cn)
            except Exception:
                continue
            paper_map[cn] = {
                "title": title,
                "authors": paper.get("authors") or [],
                "year": paper.get("year"),
                "arxiv_id": paper.get("arxiv_id") or "",
                "s2_id": paper.get("s2_id") or "",
            }
    return paper_map


# ------------------------------------------------------------------
# Section body content generation (cluster-scoped retrieval)
# ------------------------------------------------------------------

def _generate_sections(state: SurveyState, llm: LLMClient, embedder, chroma, paper_map: dict) -> dict:
    outline = state.outline
    section_content = {}

    def gen_section(level_title: tuple[int, str]) -> tuple[str, str, int] | None:
        level, title = level_title
        context = _retrieve_context(title, state, embedder, chroma, paper_map)
        if not context:
            sm = get_streaming_manager()
            sm.emit_log(state.survey_id, "writer_node",
                f"Writer Node: SKIPPING section '{title}' — "
                f"no relevant context (cluster could not be determined)"
            )
            return None
        prompt = SECTION_WRITING_USER_TEMPLATE.format(
            section_title=title,
            context=context,
        )
        sm = get_streaming_manager()

        def on_chunk(token: str):
            sm.emit_llm_token(state.survey_id, "writer_node", token)

        content = llm.generate(
            system_prompt=SECTION_WRITING_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=1024,
            temperature=0.5,
            model="deepseek-v4-pro",
            on_chunk=on_chunk,
        )
        return title, content, level

    subsections = [(l, t) for l, t in outline if l > 1]
    with concurrent.futures.ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
        futures = {
            executor.submit(gen_section, (l, t)): (l, t)
            for l, t in subsections
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result is None:
                    continue
                title, content, level = result
                section_content[(level, title)] = content
            except Exception as e:
                sm = get_streaming_manager()
                sm.emit_log(state.survey_id, "writer_node", f"Writer Node: Section generation error - {e}")

    return section_content


def _get_cluster_for_title(title: str, num_classes: int, labels: dict) -> int | None:
    """Map a section title to its cluster id.

    Cluster sections are numbered 3 .. num_classes+2 (section 1=Abstract,
    2=Introduction; the trailing macro sections Future Directions /
    Conclusion / References take num_classes+3 .. num_classes+5). The
    mapping is therefore ``cluster_id = first_digit - 3`` for any number
    of clusters, replacing the old hard-coded ``{3:0, 4:1, 5:2}`` that
    silently broke whenever the classifier produced other than 3 clusters.
    Returns None when the title carries no leading number, the number is
    outside the cluster range, or no collection carries that label.
    """
    m = re.match(r"^(\d+)", title.strip())
    if not m:
        return None
    first_digit = int(m.group(1))
    cluster_id = first_digit - 3
    if cluster_id < 0 or cluster_id >= max(num_classes, 0):
        return None
    collections_in_cluster = {
        cn for cn, lbl in labels.items()
        if lbl == cluster_id
    }
    if not collections_in_cluster:
        return None
    return cluster_id


def _get_collections_for_cluster(cluster_id: int, labels: dict) -> list[str]:
    return [cn for cn, lbl in labels.items() if lbl == cluster_id]


def _retrieve_context(
    title: str,
    state: SurveyState,
    embedder,
    chroma,
    paper_map: dict,
) -> str:
    chunks = []
    labels = getattr(state, "labels", {})
    num_classes = getattr(state, "num_classes", 0) or 0
    cluster_id = _get_cluster_for_title(title, num_classes, labels)

    if cluster_id is not None:
        target_collections = _get_collections_for_cluster(cluster_id, labels)
        state.add_log(
            f"Writer Node: Cluster-scoped retrieval for '{title}' "
            f"(cluster={cluster_id}, collections={len(target_collections)})"
        )
    else:
        # Fallback: retrieve over the WHOLE corpus rather than silently
        # dropping the section. The old code returned "" here, which
        # caused entire sections (and all their citations) to vanish
        # whenever the section number did not line up with the cluster
        # map — the root cause of surveys ending up with only 2-3
        # references. Grounding over all collections is strictly better
        # than emitting no section at all.
        target_collections = list(state.chroma_collections or labels.keys())
        state.add_log(
            f"Writer Node: No cluster match for '{title}' — "
            f"falling back to whole-corpus retrieval over "
            f"{len(target_collections)} collection(s) instead of skipping."
        )

    if not target_collections:
        state.add_log(
            f"Writer Node: No collections available at all for '{title}' "
            f"— cannot retrieve context."
        )
        return ""

    try:
        q_emb = embedder.embed_query(title)
    except Exception as e:
        state.add_log(f"Writer Node: EMBEDDING FAILED for '{title}' — {e}")
        return ""

    all_zero_distance = True
    for coll in target_collections:
        try:
            result = chroma.query(collection_name=coll, query_embeddings=[q_emb], n_results=5)
            if not isinstance(result, dict):
                state.add_log(
                    f"Writer Node: UNEXPECTED result type for '{coll}' — "
                    f"got {type(result).__name__}, expected dict."
                )
                continue
            if not result or not result.get("documents"):
                continue
            dists = result.get("distances", [[]])
            dist_list = dists[0] if dists and dists[0] else []
            for d in dist_list:
                if d is not None and abs(d) > 1e-6:
                    all_zero_distance = False

            meta_list = result.get("metadatas") or []
            doc_list = result["documents"][0] if result["documents"] else []
            for i, doc_text in enumerate(doc_list):
                chunk_index = i
                if meta_list and i < len(meta_list) and isinstance(meta_list[i], dict):
                    chunk_index = meta_list[i].get("chunk_index", i)
                source_label = _make_source_label(coll, paper_map, chunk_index)
                chunks.append(f"[{source_label}]\n{doc_text}")
        except Exception as e:
            state.add_log(
                f"Writer Node: CHROMADB QUERY FAILED for collection '{coll}' "
                f"(target='{title}') — {type(e).__name__}: {e}"
            )
            continue

    if all_zero_distance:
        state.add_log(
            f"Writer Node: ALL distances are 0.0 for '{title}' "
            f"— embedding dimension mismatch (query vs stored vectors)."
        )
        return ""

    combined = "\n\n".join(chunks[:8])
    if not combined.strip():
        state.add_log(
            f"Writer Node: No chunks retrieved for '{title}' "
            f"(target_collections={target_collections})"
        )
        return ""

    return combined[:3500]


def _make_source_label(collection_name: str, paper_map: dict, chunk_index: int) -> str:
    meta = paper_map.get(collection_name)
    if meta:
        apa_short = _format_apa_short(meta)
        return f"{apa_short} — {meta['title']} #chunk_{chunk_index}"
    readable = collection_name.replace(".", " ").replace("_", " ").strip()
    readable = " ".join(readable.split())[:80]
    return f"{readable} #chunk_{chunk_index}"


# ------------------------------------------------------------------
# Macro section generation (based on already-generated section content)
# ------------------------------------------------------------------

def _assemble_sections_for_context(section_content: dict, outline: list) -> str:
    parts = []
    for level, title in outline:
        if level == 1:
            continue
        content = section_content.get((level, title))
        if content:
            parts.append(content)
    combined = "\n\n".join(parts)
    return combined


def _llm_call(
    state: SurveyState,
    llm: LLMClient,
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float = 0.5,
) -> str:
    """Shared LLM call helper that wires up streaming callbacks."""
    sm = get_streaming_manager()

    def on_chunk(token: str):
        sm.emit_llm_token(state.survey_id, "writer_node", token)

    return llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        model=model,
        on_chunk=on_chunk,
    )


def _generate_introduction(
    state: SurveyState,
    llm: LLMClient,
    section_content: dict,
) -> str:
    topic = state.user_input
    sections_text = _assemble_sections_for_context(section_content, state.outline)
    prompt = INTRODUCTION_USER_TEMPLATE.format(
        title=topic,
        context=sections_text,
    )
    response = _llm_call(
        state, llm,
        system_prompt="You are a helpful assistant that helps generate the introduction of a survey paper.",
        user_prompt=prompt,
        model="deepseek-v4-pro",
        max_tokens=2048,
    )
    start = response.find("Introduction:")
    if start != -1:
        return response[start + len("Introduction:"):].strip()
    return response.strip()


def _generate_abstract(
    state: SurveyState,
    llm: LLMClient,
    section_content: dict,
) -> str:
    sections_text = _assemble_sections_for_context(section_content, state.outline)
    state.add_log(f"Writer Node: Abstract context length = {len(sections_text)}")
    prompt = ABSTRACT_USER_TEMPLATE.format(context=sections_text)
    try:
        response = _llm_call(
            state, llm,
            system_prompt="You are a skilled research survey writer.",
            user_prompt=prompt,
            model="deepseek-v4-pro",
            max_tokens=1024,
        )
        state.add_log(f"Writer Node: Abstract LLM response length = {len(response)}")
    except Exception as e:
        state.add_log(f"Writer Node: Abstract LLM call failed: {e}")
        return ""
    start = response.find("Abstract:")
    if start != -1:
        result = response[start + len("Abstract:"):].strip()
        state.add_log(f"Writer Node: Abstract extracted, length = {len(result)}")
        return result
    result = response.strip()
    state.add_log(f"Writer Node: Abstract no marker found, fallback length = {len(result)}")
    return result


def _generate_conclusion(
    state: SurveyState,
    llm: LLMClient,
    section_content: dict,
) -> str:
    sections_text = _assemble_sections_for_context(section_content, state.outline)
    prompt = CONCLUSION_USER_TEMPLATE.format(context=sections_text)
    response = _llm_call(
        state, llm,
        system_prompt="You are a helpful assistant that helps generate the conclusion of a survey paper.",
        user_prompt=prompt,
        model="deepseek-v4-pro",
        max_tokens=1024,
    )
    start = response.find("Conclusion:")
    if start != -1:
        return response[start + len("Conclusion:"):].strip()
    return response.strip()


def _generate_future_directions(
    state: SurveyState,
    llm: LLMClient,
    section_content: dict,
) -> str:
    body_text = _assemble_sections_for_context(section_content, state.outline)
    gap_report = getattr(state, "gap_report", "") or ""
    prompt = FUTURE_WORK_USER_TEMPLATE.format(
        body_text=body_text,
        gap_report=gap_report or "Not available",
    )
    response = _llm_call(
        state, llm,
        system_prompt="You are a helpful assistant that helps generate the future directions of a survey paper.",
        user_prompt=prompt,
        model="deepseek-v4-pro",
        max_tokens=1024,
    )
    start = response.find("Future Directions:")
    if start != -1:
        return response[start + len("Future Directions:"):].strip()
    return response.strip()


def _read_assembled_survey(md_file_path: str) -> str:
    try:
        return Path(md_file_path).read_text(encoding="utf-8")
    except Exception:
        return ""


def _generate_abstract_from_assembled(
    state: SurveyState,
    llm: LLMClient,
    assembled_text: str,
) -> str:
    state.add_log(f"Writer Node: Abstract assembled text length = {len(assembled_text)}")
    prompt = ABSTRACT_USER_TEMPLATE.format(context=assembled_text)
    try:
        response = _llm_call(
            state, llm,
            system_prompt="You are a skilled research survey writer.",
            user_prompt=prompt,
            model="deepseek-v4-pro",
            max_tokens=1024,
        )
        state.add_log(f"Writer Node: Abstract re-generation response length = {len(response)}")
    except Exception as e:
        state.add_log(f"Writer Node: Abstract re-generation failed: {e}")
        return getattr(state, "abstract", "")
    start = response.find("Abstract:")
    if start != -1:
        return response[start + len("Abstract:"):].strip()
    return response.strip()


def _generate_introduction_from_assembled(
    state: SurveyState,
    llm: LLMClient,
    assembled_text: str,
) -> str:
    topic = state.user_input
    prompt = INTRODUCTION_USER_TEMPLATE.format(title=topic, context=assembled_text)
    response = _llm_call(
        state, llm,
        system_prompt="You are a helpful assistant that helps generate the introduction of a survey paper.",
        user_prompt=prompt,
        model="deepseek-v4-pro",
        max_tokens=2048,
    )
    start = response.find("Introduction:")
    if start != -1:
        return response[start + len("Introduction:"):].strip()
    return response.strip()


def _generate_conclusion_from_assembled(
    state: SurveyState,
    llm: LLMClient,
    assembled_text: str,
) -> str:
    prompt = CONCLUSION_USER_TEMPLATE.format(context=assembled_text)
    response = _llm_call(
        state, llm,
        system_prompt="You are a helpful assistant that helps generate the conclusion of a survey paper.",
        user_prompt=prompt,
        model="deepseek-v4-pro",
        max_tokens=1024,
    )
    start = response.find("Conclusion:")
    if start != -1:
        return response[start + len("Conclusion:"):].strip()
    return response.strip()


def _generate_future_from_assembled(
    state: SurveyState,
    llm: LLMClient,
    assembled_text: str,
) -> str:
    gap_report = getattr(state, "gap_report", "") or ""
    prompt = FUTURE_WORK_USER_TEMPLATE.format(
        body_text=assembled_text,
        gap_report=gap_report or "Not available",
    )
    response = _llm_call(
        state, llm,
        system_prompt="You are a helpful assistant that helps generate the future directions of a survey paper.",
        user_prompt=prompt,
        model="deepseek-v4-pro",
        max_tokens=1024,
    )
    start = response.find("Future Directions:")
    if start != -1:
        return response[start + len("Future Directions:"):].strip()
    return response.strip()


def _format_authors_apa(authors: list[str]) -> str:
    if not authors:
        return "Unknown Author"

    def _format_one(name: str) -> str:
        parts = name.strip().split()
        if len(parts) == 1:
            return parts[0]
        last = parts[-1]
        initials = " ".join(f"{p[0]}." for p in parts[:-1])
        return f"{last}, {initials}"

    formatted = [_format_one(a) for a in authors]
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]}, & {formatted[1]}"
    if len(formatted) <= 20:
        return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"
    return ", ".join(formatted[:19]) + ", ... " + formatted[-1]


def _generate_references(state: SurveyState, llm: LLMClient, paper_meta_map: dict) -> str:
    meta = paper_meta_map or state.citation_paper_meta
    lines = []
    seen_keys: set = set()
    for cn, info in meta.items():
        authors = info.get("authors") or []
        title = info.get("title", "Unknown Title")
        year = info.get("year")
        year_str = str(year) if year else "n.d."
        arxiv_id = (info.get("arxiv_id") or "").strip().rstrip("vV")

        title = title.rstrip(".:")
        author_str = _format_authors_apa(authors)

        if arxiv_id:
            arxiv_str = f"arXiv:{arxiv_id}"
            ref = f"{author_str} ({year_str}). {title}. {arxiv_str}."
        else:
            ref = f"{author_str} ({year_str}). {title}."

        dedup_key = arxiv_id.lower() if arxiv_id else f"title::{title.lower()}"
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        lines.append(f"Reference: {ref}")

    return "\n\n".join(lines)


# ------------------------------------------------------------------
# Assembly
# ------------------------------------------------------------------

def _clean_section_content(content: str) -> str:
    lines = content.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^#{1,6}\s+\S", stripped):
            continue
        if stripped.lower().startswith("here is the detailed"):
            continue
        if stripped.lower().startswith("here is the survey paper"):
            continue
        if stripped.lower().startswith("survey paper content for"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _assemble_and_save(state: SurveyState, section_content: dict, paper_meta_map: dict) -> tuple[str, str]:
    parts = []

    for level, title in state.outline:
        prefix = "#" * level
        parts.append(f"{prefix} {title}\n")
        if (level, title) in section_content:
            content = _clean_section_content(section_content[(level, title)])
            parts.append(content + "\n")

    for section_name, content in [
        ("Abstract", state.abstract),
        ("Introduction", state.introduction),
        ("Future Directions", state.future_directions),
        ("Conclusion", state.conclusion),
        ("References", state.references),
    ]:
        if content:
            for i, line in enumerate(parts):
                pattern = rf"^#+\s+\d*\.?\s*{re.escape(section_name)}\s*$"
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    insert_pos = parts.index(line) + 1
                    parts.insert(insert_pos, content + "\n\n")
                    break

    survey_text = "\n".join(parts)

    survey_dir = Path("./data/md") / state.survey_id
    survey_dir.mkdir(parents=True, exist_ok=True)
    md_file_path = survey_dir / "survey.md"
    md_file_path.write_text(survey_text, encoding="utf-8")
    state.add_log(f"Writer Node: Saved md file to {md_file_path}")

    return str(md_file_path), survey_text

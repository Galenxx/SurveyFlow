"""Shared citation-writing helpers for the benchmark generators.

These mirror the citation surface of ``nodes.writer_node`` (APA source
labels and APA reference lines) so that NAIVE_RAG and any other ablation
arm emit citations in the EXACT same format SurveyFlow does. Keeping the
format identical is what makes the cross-system citation-quality
comparison fair: any measured difference is attributable to retrieval /
structure, never to a different reference style.

The functions are intentionally pure (no streaming, no state mutation) so
they can be reused outside the LangGraph node context.
"""

from __future__ import annotations


# --- APA author / short-label formatting (mirrors writer_node) ----------


def format_apa_short(meta: dict) -> str:
    """APA in-text short form, e.g. 'Frej et al. (2024)'. Mirrors
    ``writer_node._format_apa_short`` exactly."""
    authors: list = meta.get("authors", []) or []
    year = meta.get("year") or "n.d."
    year_str = str(year) if year != "n.d." else "n.d."

    if not authors:
        title = meta.get("title", "")
        truncated = title[:60].strip()
        return f"{truncated} ({year_str})"

    if len(authors) == 1:
        first = authors[0].split()[-1]
        return f"{first} ({year_str})"
    if len(authors) == 2:
        a1 = authors[0].split()[-1]
        a2 = authors[1].split()[-1]
        return f"{a1} & {a2} ({year_str})"
    first = authors[0].split()[-1]
    return f"{first} et al. ({year_str})"


def make_source_label(collection_name: str, paper_map: dict, chunk_index: int) -> str:
    """Build the bracketed source label prepended to each retrieved chunk.
    Mirrors ``writer_node._make_source_label``."""
    meta = paper_map.get(collection_name)
    if meta:
        apa_short = format_apa_short(meta)
        return f"{apa_short} — {meta['title']} #chunk_{chunk_index}"
    readable = collection_name.replace(".", " ").replace("_", " ").strip()
    readable = " ".join(readable.split())[:80]
    return f"{readable} #chunk_{chunk_index}"


def _format_authors_apa(authors: list) -> str:
    """APA author list for a reference line. Mirrors
    ``writer_node._format_authors_apa``."""
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


def assemble_references(used_collections, paper_map: dict) -> str:
    """Build the 'Reference:' block from the papers actually grounded in.

    ``used_collections`` is the set of ChromaDB collection names that
    surfaced at least one chunk during writing. Only those papers are
    cited — exactly mirroring SurveyFlow's "references from used papers"
    contract. Output format is identical to
    ``writer_node._generate_references``.
    """
    lines: list[str] = []
    seen_keys: set = set()
    for cn in used_collections:
        info = paper_map.get(cn)
        if not info:
            continue
        authors = info.get("authors") or []
        title = (info.get("title") or "Unknown Title").rstrip(".:")
        year = info.get("year")
        year_str = str(year) if year else "n.d."
        arxiv_id = (info.get("arxiv_id") or "").strip().rstrip("vV")

        author_str = _format_authors_apa(authors)
        if arxiv_id:
            ref = f"{author_str} ({year_str}). {title}. arXiv:{arxiv_id}."
        else:
            ref = f"{author_str} ({year_str}). {title}."

        dedup_key = arxiv_id.lower() if arxiv_id else f"title::{title.lower()}"
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        lines.append(f"Reference: {ref}")

    return "\n\n".join(lines)


def count_reference_lines(text: str) -> int:
    """Count 'Reference:' lines — the same surface the parser keys on."""
    return sum(
        1 for line in text.splitlines()
        if line.strip().lower().startswith("reference:")
    )


def _strip_markup(s: str) -> str:
    """Remove leading markdown heading hashes and surrounding bold/italic
    markers so a self-repeated heading can be compared to its title in any
    surface form (``### 3 Background``, ``**3 Background**``, ``3 Background``)."""
    s = s.strip()
    s = s.lstrip("#").strip()
    # strip matched ** ** or * * or __ __ wrappers (possibly repeated)
    for marker in ("**", "__", "*", "_"):
        while s.startswith(marker) and s.endswith(marker) and len(s) > 2 * len(marker):
            s = s[len(marker):-len(marker)].strip()
    return s


def clean_section_content(content: str, section_title: str | None = None) -> str:
    """Strip the LLM's self-repeated section headings and boilerplate
    preambles from generated section body text.

    Supersets ``writer_node._clean_section_content``: that node-side cleaner
    only drops markdown ``#`` headings, which leaves the *other* two forms the
    section writer LLM emits — a fully-bold line (``**5 Applications**``) or a
    bare numbered title (``6 Challenges and Open Problems``) — sitting in the
    body. Those stray heading-repeats matter here because the benchmark's
    sentence-level attribution (Task #8) would otherwise treat each as an
    uncited "sentence" and unfairly count it as a bare assertion.

    We drop a line when:
      * it is a markdown heading (``#``..``######`` + text),
      * it is a known boilerplate preamble ("Here is the detailed..."), or
      * (when ``section_title`` is given) its markup-stripped form equals the
        section title — catching the bold/plain heading-repeats in any form
        with zero risk of deleting real prose (an exact title match only).

    NOTE: for cross-system fairness the SAME cleaner must run on SurveyFlow's
    output at parse/attribute time, since SurveyFlow's node-side cleaner is the
    weaker ``#``-only variant. Recorded in the v2 benchmark memory.

    Kept pure (no state) so both the node and the benchmark generators can
    share one implementation.
    """
    import re as _re

    title_norm = _strip_markup(section_title).lower() if section_title else None

    lines = content.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if _re.match(r"^#{1,6}\s+\S", stripped):
            continue
        low = stripped.lower()
        if low.startswith("here is the detailed"):
            continue
        if low.startswith("here is the survey paper"):
            continue
        if low.startswith("survey paper content for"):
            continue
        if title_norm and _strip_markup(stripped).lower() == title_norm:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

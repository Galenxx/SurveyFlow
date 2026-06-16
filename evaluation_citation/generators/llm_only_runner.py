"""LLM-only (no-retrieval) generator runner.

This is the "parametric knowledge" baseline. The same LLM as
SurveyFlow (default glm-5.1) is asked to write a full survey on
`topic` WITHOUT any retrieval context. The prompt forces it to
cite ~10 real papers (this is the exact thing being tested - the
LLM has to remember and cite real papers, which is hard).
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

from evaluation_citation.configs import (
    CITATIONS_PER_SURVEY,
    LLM_ONLY_MODEL,
    SURVEY_OUTPUT_DIR,
)


# SurveyFlow writer prompts typically render in the format:
#   # 1 Abstract
#   <text>
#   # 2 Introduction
#   <text>
#   ...
#   Reference: Author, X. (2024). Title. arXiv:NNNN.NNNNN.
# To make the two systems directly comparable, the LLM_ONLY prompt
# asks the model to produce a survey in the same structural style.
_PROMPT = """You are an expert researcher writing a survey paper.

Topic: {topic}

Write a complete survey in Markdown. The structure MUST follow exactly:

# 1 Abstract
<200-250 word abstract>

# 2 Introduction
<500-700 word introduction covering background, motivation, scope, structure>

# 3 <Section 1 title>
<content with 1-2 in-text (Author, Year) citations>

# 4 <Section 2 title>
<content with 1-2 in-text (Author, Year) citations>

# 5 <Section 3 title>
<content with 1-2 in-text (Author, Year) citations>

# 6 <Section 4 title>
<content with 1-2 in-text (Author, Year) citations>

# 7 Conclusion
<300-500 word conclusion>

# 8 Future Directions
<300-500 word future directions>

Then a "References" section containing exactly {n_citations} references,
each in this format on its own line:

Reference: <Authors>. (<Year>). <Full Paper Title>. arXiv:<arxiv_id>v<n>.

Requirements for the references:
- The references must be REAL papers that exist on arXiv.
- Cite the paper's actual authors, year, title, and arXiv ID.
- Do not invent or hallucinate references - if you are not confident
  a paper is real, do not include it.
- The total number of references MUST be exactly {n_citations}.

Begin writing the survey now. Output ONLY the survey markdown, no preamble."""


def _write_survey_md(out_path: Path, body: str, topic: str) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    return out_path


def _force_n_references(body: str, n: int) -> str:
    """Best-effort trim: keep only the first n 'Reference:' lines."""
    lines = body.splitlines()
    kept: list[str] = []
    ref_count = 0
    seen_refs_header = False
    for line in lines:
        if re.match(r"^\s*#{1,6}\s+References\s*$", line, re.IGNORECASE):
            seen_refs_header = True
            kept.append(line)
            continue
        if re.match(r"^\s*Reference:\s*", line, re.IGNORECASE):
            if ref_count >= n:
                continue
            ref_count += 1
            kept.append(line)
        else:
            kept.append(line)
    if not seen_refs_header:
        # No references header at all - append a stub so the parser
        # can still extract the references.
        if ref_count < n:
            kept.append("")
            kept.append("# References")
        for i in range(ref_count, n):
            kept.append(
                f"Reference: Unknown Author. (2024). Placeholder Reference {i + 1}. "
                f"arXiv:0000.00000v1."
            )
    return "\n".join(kept)


def run_llm_only(
    topic: str,
    *,
    paper_count: int = CITATIONS_PER_SURVEY,
    model: str = LLM_ONLY_MODEL,
    seed: Optional[int] = None,
) -> str:
    """Generate a survey for `topic` using the LLM only (no retrieval).

    Returns the path to the saved survey.md (back-compat thin wrapper).
    """
    return run_llm_only_detailed(
        topic, paper_count=paper_count, model=model, seed=seed
    )["md_path"]


def run_llm_only_detailed(
    topic: str,
    *,
    paper_count: int = CITATIONS_PER_SURVEY,
    model: str = LLM_ONLY_MODEL,
    seed: Optional[int] = None,
) -> dict:
    """Generate a survey for `topic` using the LLM only (no retrieval) and
    return md_path + generation-guard telemetry, mirroring the contract of
    ``run_surveyflow_detailed`` / ``run_naive_rag_detailed``::

        {
          "md_path": str,
          "survey_id": "",        # LLM_ONLY has no pipeline survey_id
          "n_references_realized": int,
          "corpus_stats": {...},  # all zero — LLM_ONLY has NO corpus
        }

    For LLM_ONLY the corpus is empty by construction (parametric knowledge
    only), so corpus_stats are recorded as zero. The realized reference
    count is forced to ``paper_count`` by ``_force_n_references`` but we
    count the actual surface lines anyway, so a degenerate generation is
    still visible in the telemetry.
    """
    from core.llm_client import LLMClient

    prompt = _PROMPT.format(topic=topic, n_citations=paper_count)
    llm = LLMClient()
    body = llm.generate(
        system_prompt="You are a careful academic survey writer. Only cite papers you are confident are real.",
        user_prompt=prompt,
        max_tokens=16384,
        temperature=0.5,
        model=model,
    )
    body = _force_n_references(body, paper_count)

    out_dir = SURVEY_OUTPUT_DIR / "llm_only"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^a-zA-Z0-9_]+", "_", topic[:50]).strip("_")
    seed_tag = f"__seed{seed}" if seed is not None else ""
    out_path = out_dir / f"{safe_id}__{model}{seed_tag}.md"
    _write_survey_md(out_path, body, topic)

    n_realized = sum(
        1 for line in body.splitlines()
        if line.strip().lower().startswith("reference:")
    )
    return {
        "md_path": str(out_path),
        "survey_id": "",
        "n_references_realized": n_realized,
        "corpus_stats": {
            "n_papers_in_list": 0,
            "n_collections_embedded": 0,
            "n_citation_meta": 0,
            "total_chunks": 0,
            "note": "LLM_ONLY has no retrieval corpus by construction",
        },
    }


if __name__ == "__main__":  # pragma: no cover
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m evaluation_citation.generators.llm_only_runner <topic>")
        sys.exit(1)
    path = run_llm_only(sys.argv[1])
    print(f"saved: {path}")

"""Citation parser for the Citation Accuracy benchmark.

SurveyFlow produces Markdown surveys whose reference section is a list of
lines starting with "Reference:" (no [n] numbering, no author-year
in-text anchors are required to be machine-parseable). This module
extracts:

  1. The survey topic (from H1 / first heading, or from metadata).
  2. The list of references - one dict per "Reference:" line, with
     best-effort structured fields (title, authors, year, arxiv_id,
     doi) extracted via regex.

The parser is intentionally permissive: it never throws on a malformed
reference, it just returns whatever fields it could extract. The
verifier layer (verifiers/...) is responsible for resolving any
unparsed fields via API lookup.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# --- regex patterns -----------------------------------------------------

# A reference line in the SurveyFlow output looks like:
#   Reference: Kan, L., Yu, Z., & Liu, X. (2026). JobRec: A Dual-Perspective
#       Recommendation Framework. arXiv:2601.12345v2.
_REFERENCE_LINE_RE = re.compile(
    r"^\s*Reference:\s*(.+?)\s*$",
    re.IGNORECASE,
)

# arXiv ID in either modern (YYMM.NNNNN) or old (cs.LG/0703001) format,
# possibly with a version suffix.
_ARXIV_ID_RE = re.compile(
    r"arXiv:\s*((?:\d{4}\.\d{4,5}(?:v\d+)?)|(?:[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?))",
    re.IGNORECASE,
)

_DOI_RE = re.compile(r"\bdoi:\s*(\S+)", re.IGNORECASE)

# Try to extract the publication year. We accept any 4-digit year
# 1980-2099 surrounded by word boundaries that is NOT obviously an
# arXiv id segment.
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")

# Extract the title - text between ". (YYYY)." and "arXiv:" / "doi:".
_TITLE_AFTER_YEAR_RE = re.compile(
    r"\(\s*(?:19|20)\d{2}\s*\)\.\s*(.+?)(?=\s*(?:arXiv:|doi:|$))",
    re.IGNORECASE,
)


@dataclass
class ParsedReference:
    """A single reference extracted from a SurveyFlow-style survey."""

    ref_id: int                          # 0-based index in the reference list
    raw: str                             # full reference line text
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[str] = None
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParsedSurvey:
    """Result of parsing one survey markdown file."""

    path: str
    topic: Optional[str]
    n_references: int
    references: list[ParsedReference] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "topic": self.topic,
            "n_references": self.n_references,
            "references": [r.to_dict() for r in self.references],
        }


# --- public API ---------------------------------------------------------


def _extract_title(raw: str) -> Optional[str]:
    """Best-effort title extraction from a reference line.

    Strategy: take the text after the year+parenthesis pattern and
    before the first arXiv/DOI marker. Trim trailing punctuation.
    """
    m = _TITLE_AFTER_YEAR_RE.search(raw)
    if not m:
        return None
    title = m.group(1).strip().rstrip(".,;")
    # collapse internal whitespace
    title = re.sub(r"\s+", " ", title)
    return title or None


def _extract_authors(raw: str) -> Optional[str]:
    """Authors are the text between the leading "Reference:" and the
    first parenthesized year."""
    m = re.search(r"^[^()]*?(?=\s*\(\s*(?:19|20)\d{2}\s*\))", raw)
    if not m:
        return None
    s = m.group(0).strip().rstrip(",. ")
    return s or None


def _extract_year(raw: str) -> Optional[str]:
    m = _YEAR_RE.search(raw)
    if not m:
        return None
    y = int(m.group(1))
    if 1980 <= y <= 2099:
        return m.group(1)
    return None


def _extract_arxiv_id(raw: str) -> Optional[str]:
    m = _ARXIV_ID_RE.search(raw)
    return m.group(1) if m else None


def _extract_doi(raw: str) -> Optional[str]:
    m = _DOI_RE.search(raw)
    if not m:
        return None
    return m.group(1).rstrip(".,;)")


def _extract_topic(text: str, fallback: Optional[str] = None) -> Optional[str]:
    """Heuristic: the first H1 heading that is NOT a section label
    (e.g. "# 1 Abstract", "# 2 Introduction", "# References") is
    treated as the survey title. If none is found, falls back to the
    first H1 or to the caller-supplied `fallback` argument.
    """
    section_keywords = {
        "abstract", "introduction", "background", "related work",
        "methodology", "methods", "results", "discussion",
        "conclusion", "conclusions", "future work", "future directions",
        "acknowledgements", "acknowledgments", "references",
        "appendix", "evaluation", "experiments",
    }
    h1s: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^#\s+(.+?)\s*$", line)
        if m:
            h1s.append(m.group(1).strip())
    for h1 in h1s:
        # strip a leading numbering like "1", "1.", "1)", "Section 1"
        clean = re.sub(r"^(?:section\s+)?\d+\s*[\.\)]?\s*", "", h1, flags=re.IGNORECASE).strip()
        # If after stripping, the remaining text is itself a section
        # keyword, treat this H1 as a section label and keep looking.
        if clean.lower() in section_keywords:
            continue
        return h1
    if h1s:
        return h1s[0]
    return fallback


def parse_survey_md(md_path: str | Path, topic_hint: Optional[str] = None) -> ParsedSurvey:
    """Parse one survey markdown file into structured references.

    Returns a ParsedSurvey. Never throws; missing fields stay None.
    The `topic_hint` is used as a fallback when the survey's H1 headings
    are all section labels (e.g. "# 1 Abstract", "# 2 Introduction").
    """
    p = Path(md_path)
    text = p.read_text(encoding="utf-8", errors="replace")
    topic = _extract_topic(text, fallback=topic_hint)
    refs: list[ParsedReference] = []

    for line in text.splitlines():
        m = _REFERENCE_LINE_RE.match(line)
        if not m:
            continue
        raw = m.group(1).strip()
        refs.append(
            ParsedReference(
                ref_id=len(refs),
                raw=raw,
                title=_extract_title(raw),
                authors=_extract_authors(raw),
                year=_extract_year(raw),
                arxiv_id=_extract_arxiv_id(raw),
                doi=_extract_doi(raw),
            )
        )

    return ParsedSurvey(
        path=str(p),
        topic=topic,
        n_references=len(refs),
        references=refs,
    )


def parse_payload_payload(parsed: ParsedSurvey) -> dict:
    """Convenience wrapper that returns the JSON-serializable dict."""
    return parsed.to_dict()


# --- CLI helper ---------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m evaluation_citation.citation_parser <md_path>")
        sys.exit(1)
    out = parse_survey_md(sys.argv[1])
    print(json.dumps(out.to_dict(), indent=2, ensure_ascii=False))

"""Sentence-level attribution parser (ALCE-style) — Task #8 core.

This is the parser half of the v2 benchmark's *main* metric. Where the v1
``citation_parser`` only extracted the trailing ``Reference:`` list, this
module reads the survey BODY and answers, for every factual sentence:

  * which in-text citations (if any) are attached to it, and
  * which parsed reference each in-text citation resolves to.

That ((sentence) -> (cited papers)) mapping is what lets the attribution
judge (§4.D) ask the non-circular question "does the cited evidence
actually support THIS sentence?" instead of v1's circular "does the
reference exist?".

Design notes
------------
* In-text citations follow the APA forms the writer prompts enforce
  (``writer_prompts.SECTION_WRITING_USER_TEMPLATE``):
    parenthetical : ``(Frej, 2024)`` ``(Frej & Dai, 2024)`` ``(Frej et al., 2024)``
    narrative     : ``Frej (2024)`` ``Frej and Dai (2024)`` ``Frej et al. (2024)``
* Resolution to a reference is by (first-author last name, year). The
  ``Reference:`` lines already carry parsed authors+year (via
  ``citation_parser``), so we match against those — no network calls here.
* "Factual sentence" is approximated structurally (a sentence in a BODY
  section, long enough to assert something, not a heading/list scaffold).
  The judge makes the final call on factuality; this parser only needs to
  not throw away real claims.

The parser is permissive and never throws on malformed input — it returns
whatever structure it could recover, mirroring ``citation_parser``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from evaluation_citation.citation_parser import parse_survey_md
from evaluation_citation.generators._writing import clean_section_content


# --- in-text citation regexes ------------------------------------------

# Parenthetical group: one or more citations inside a single (...). We
# capture the inner text and split on ';' for multi-citation parentheticals
# like "(Frej, 2024; Dai, 2023)".
_PAREN_GROUP_RE = re.compile(r"\(([^()]*?\d{4}[a-z]?[^()]*?)\)")

# A single parenthetical citation's inner form:
#   "Frej, 2024" / "Frej & Dai, 2024" / "Frej et al., 2024"
_PAREN_CITE_RE = re.compile(
    r"^\s*(?P<authors>[A-Z][^,]*?(?:\s+et\s+al\.?|(?:\s*&\s*[A-Z][^,]*?))?)\s*,\s*"
    r"(?P<year>\d{4})[a-z]?\s*$"
)

# Narrative citation:  "Frej (2024)" / "Frej and Dai (2024)" / "Frej et al. (2024)"
# Author chunk stops before the "(YYYY)".
_NARRATIVE_RE = re.compile(
    r"(?P<authors>[A-Z][A-Za-z'\-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z'\-]+|\s+et\s+al\.?)?)"
    r"\s+\((?P<year>\d{4})[a-z]?\)"
)


# --- data structures ----------------------------------------------------


@dataclass
class InTextCite:
    """One in-text citation occurrence and what it resolved to."""

    surface: str                         # raw matched text, e.g. "(Frej et al., 2024)"
    authors_text: str                    # extracted author chunk, e.g. "Frej et al."
    first_author: str                    # normalized first-author last name
    year: Optional[str]
    style: str                           # "parenthetical" | "narrative"
    resolved_ref_id: Optional[int] = None  # index into the survey's reference list

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AttributedSentence:
    """A body sentence with its attached in-text citations."""

    sent_id: int
    section_title: str
    text: str
    cites: list[InTextCite] = field(default_factory=list)
    is_candidate_claim: bool = True      # structural factual-claim heuristic

    @property
    def has_citation(self) -> bool:
        return len(self.cites) > 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["has_citation"] = self.has_citation
        return d


@dataclass
class AttributionDoc:
    """Full sentence-level attribution view of one survey."""

    path: str
    topic: Optional[str]
    n_sentences: int
    n_claims: int                        # candidate factual claims
    n_claims_with_cite: int
    n_claims_bare: int
    sentences: list[AttributedSentence] = field(default_factory=list)
    references: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "topic": self.topic,
            "n_sentences": self.n_sentences,
            "n_claims": self.n_claims,
            "n_claims_with_cite": self.n_claims_with_cite,
            "n_claims_bare": self.n_claims_bare,
            "sentences": [s.to_dict() for s in self.sentences],
            "references": self.references,
        }


# --- author-name normalization (mirror identifier_check semantics) ------


def _last_name(author_chunk: str) -> str:
    """Best-effort first-author last name from an in-text author chunk.

    Handles "Frej", "Frej & Dai", "Frej et al.", "Frej and Dai". For
    reference-side ``authors`` strings like "Pornprasit, C., &
    Tantithamthavorn, C." the first token before the comma is the surname.
    """
    if not author_chunk:
        return ""
    s = author_chunk.strip()
    # cut at the first author boundary
    s = re.split(r"\s+et\s+al\.?|\s*&\s*|\s+and\s+", s)[0].strip()
    # "Lastname, F." -> "Lastname"
    if "," in s:
        s = s.split(",", 1)[0].strip()
    parts = s.split()
    return parts[-1].strip().lower() if parts else ""


# --- body extraction ----------------------------------------------------


# Headings whose bodies are citation-free by writer-prompt construction —
# we still scan them (a stray citation there is worth surfacing) but mark
# their sentences as non-claims so they don't inflate the bare-assertion
# denominator.
_NON_CLAIM_SECTIONS = {
    "abstract", "introduction", "conclusion",
    "future directions", "future work", "references",
}


def _heading_key(title: str) -> str:
    """Normalize a heading like '3 Background and Foundations' for matching:
    strip leading number and lowercase."""
    t = re.sub(r"^\s*#+\s*", "", title)            # drop markdown hashes
    t = re.sub(r"^\s*\d+(?:\.\d+)*\s*", "", t)      # drop leading numbering
    return t.strip().lower()


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter.

    Splits on ., !, ? followed by whitespace+capital/EOL, while trying not
    to break on common abbreviations ("et al.", "e.g.", "i.e.", "Fig.",
    decimals). Good enough for claim counting; the judge sees full context.
    """
    # protect abbreviations and decimals
    protected = text
    protections = {
        r"\bet al\.": "et␞al␞",
        r"\be\.g\.": "e␞g␞",
        r"\bi\.e\.": "i␞e␞",
        r"\bcf\.": "cf␞",
        r"\bFig\.": "Fig␞",
        r"\bvs\.": "vs␞",
        r"\bal\.": "al␞",
    }
    for pat, repl in protections.items():
        protected = re.sub(pat, repl, protected)
    # protect decimals like "3.5"
    protected = re.sub(r"(\d)\.(\d)", r"\1␞\2", protected)

    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", protected)
    out = []
    for p in pieces:
        s = p.replace("␞", ".").strip()  # crude un-protect (dot back)
        # fix the decimal/abbrev un-protect: "et␞al␞" -> "et al."
        s = s.replace("et.al.", "et al.").replace("e.g.", "e.g.").replace("i.e.", "i.e.")
        if s:
            out.append(s)
    return out


def _extract_intext_cites(sentence: str) -> list[InTextCite]:
    """Find all in-text citations in a sentence (both APA styles)."""
    cites: list[InTextCite] = []

    # 1. parenthetical groups (may contain ';'-separated multiples)
    for gm in _PAREN_GROUP_RE.finditer(sentence):
        inner = gm.group(1)
        for piece in inner.split(";"):
            m = _PAREN_CITE_RE.match(piece.strip())
            if not m:
                continue
            authors_text = m.group("authors").strip()
            cites.append(InTextCite(
                surface=gm.group(0),
                authors_text=authors_text,
                first_author=_last_name(authors_text),
                year=m.group("year"),
                style="parenthetical",
            ))

    # 2. narrative citations (avoid double-counting ones already inside
    #    a parenthetical, which the narrative regex won't match anyway
    #    because it requires "Author (YYYY)" not "(Author, YYYY)").
    for m in _NARRATIVE_RE.finditer(sentence):
        authors_text = m.group("authors").strip()
        cites.append(InTextCite(
            surface=m.group(0),
            authors_text=authors_text,
            first_author=_last_name(authors_text),
            year=m.group("year"),
            style="narrative",
        ))

    return cites


def _resolve_cite_to_ref(cite: InTextCite, references: list[dict]) -> Optional[int]:
    """Match an in-text citation to a parsed reference by (last name, year).

    Year is matched within +/-1 (preprint vs publication drift), surname by
    exact normalized equality. Returns the reference's ``ref_id`` or None.
    """
    if not cite.first_author:
        return None
    best: Optional[int] = None
    for ref in references:
        ref_last = _last_name(ref.get("authors") or "")
        if ref_last != cite.first_author:
            continue
        ref_year = ref.get("year")
        if cite.year and ref_year:
            try:
                if abs(int(cite.year) - int(ref_year)) > 1:
                    continue
            except (TypeError, ValueError):
                pass
        best = ref.get("ref_id")
        break
    return best


# --- public API ---------------------------------------------------------


def parse_attribution(md_path: str | Path, topic_hint: Optional[str] = None) -> AttributionDoc:
    """Build the sentence-level attribution view of a survey.

    Reuses ``citation_parser.parse_survey_md`` for the reference list +
    topic, then walks the body section-by-section extracting sentences and
    their in-text citations and resolving each citation to a reference.
    """
    p = Path(md_path)
    text = p.read_text(encoding="utf-8", errors="replace")

    parsed = parse_survey_md(md_path, topic_hint=topic_hint)
    references = [r.to_dict() for r in parsed.references]

    sentences: list[AttributedSentence] = []
    sent_counter = 0

    # Walk the doc heading-by-heading. A heading line is "# ... " (any
    # level). Everything until the next heading is that section's body.
    current_heading = ""
    current_key = ""
    body_buf: list[str] = []

    def flush_section():
        nonlocal sent_counter
        if not current_heading:
            return
        body_text = "\n".join(body_buf).strip()
        if not body_text:
            return
        # Cross-system fairness (recorded bias hazard): SurveyFlow's writer
        # node only ran the weaker #-only cleaner, so its .md can carry
        # self-repeated bold/bare heading lines that would otherwise be
        # counted as uncited sentences and inflate BareAssertionRate. We run
        # the SAME strong cleaner here on EVERY system's body at parse time so
        # the attribution view is symmetric (idempotent on NAIVE_RAG, which
        # already cleaned at write time). section_title enables exact
        # heading-echo removal in any markup form.
        body_text = clean_section_content(body_text, section_title=current_heading).strip()
        if not body_text:
            return
        is_claim_section = current_key not in _NON_CLAIM_SECTIONS
        for raw_sent in _split_sentences(body_text):
            # skip stray reference lines / list scaffolding
            if raw_sent.lower().startswith("reference:"):
                continue
            cites = _extract_intext_cites(raw_sent)
            for c in cites:
                c.resolved_ref_id = _resolve_cite_to_ref(c, references)
            # structural claim heuristic: long enough, in a claim section,
            # not a pure heading echo
            is_candidate = (
                is_claim_section
                and len(raw_sent.split()) >= 6
            )
            sentences.append(AttributedSentence(
                sent_id=sent_counter,
                section_title=current_heading,
                text=raw_sent,
                cites=cites,
                is_candidate_claim=is_candidate,
            ))
            sent_counter += 1

    for line in text.splitlines():
        if re.match(r"^#+\s+\S", line):
            flush_section()
            current_heading = re.sub(r"^#+\s+", "", line).strip()
            current_key = _heading_key(current_heading)
            body_buf = []
        else:
            body_buf.append(line)
    flush_section()

    claims = [s for s in sentences if s.is_candidate_claim]
    claims_with_cite = [s for s in claims if s.has_citation]
    claims_bare = [s for s in claims if not s.has_citation]

    return AttributionDoc(
        path=str(p),
        topic=parsed.topic,
        n_sentences=len(sentences),
        n_claims=len(claims),
        n_claims_with_cite=len(claims_with_cite),
        n_claims_bare=len(claims_bare),
        sentences=sentences,
        references=references,
    )


# --- CLI helper ---------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import json
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m evaluation_citation.attribution_parser <md_path>")
        sys.exit(1)
    doc = parse_attribution(sys.argv[1])
    out = doc.to_dict()
    # print a compact summary, not the whole sentence dump
    print(json.dumps({
        "topic": out["topic"],
        "n_sentences": out["n_sentences"],
        "n_claims": out["n_claims"],
        "n_claims_with_cite": out["n_claims_with_cite"],
        "n_claims_bare": out["n_claims_bare"],
        "n_references": len(out["references"]),
        "sample_claims_with_cite": [
            {"text": s["text"][:120], "cites": [c["surface"] for c in s["cites"]],
             "resolved": [c["resolved_ref_id"] for c in s["cites"]]}
            for s in out["sentences"] if s["has_citation"]
        ][:5],
    }, indent=2, ensure_ascii=False))

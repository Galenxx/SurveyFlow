"""Existence-failure classification (Benchmark v2, design doc §4.B).

v1 reported a single "existence rate" that the research critique flagged
as near-circular: SurveyFlow only cites papers it downloaded and embedded,
so by construction they exist. v2 **downgrades** existence from the main
metric to a *pipeline-correctness gate* and, crucially, **decomposes every
failure** into a diagnostic class. The point is to show that SurveyFlow's
residual failures are engineering bugs (parse/id/dedup), NOT hallucination,
while LLM_ONLY's failures concentrate in genuine fabrication.

Four failure classes (§4.B):

  - METADATA_PARSE_ERROR   author / title / year could not be extracted
                           from the reference line at all -> the reference
                           string itself is malformed, so resolution never
                           had a fair chance.
  - ID_EXTRACTION_FAIL     a usable title exists, but no arXiv id / DOI was
                           extractable AND the title-only lookup failed to
                           land an unambiguous hit (AMBIGUOUS / NOT_FOUND).
                           The paper may well exist; our identifier surface
                           was too thin to pin it.
  - DEDUP_COLLISION        an id WAS given and the paper exists, but the id
                           resolves to a *different* paper than the title
                           describes (verdict MISMATCH). In SurveyFlow this
                           is the dedup step collapsing two distinct papers
                           onto one id, not a hallucination.
  - GENUINELY_FABRICATED   no candidate exists anywhere (NOT_FOUND) AND the
                           metadata was well-formed enough that a real paper
                           *should* have been found -> the generator likely
                           invented it. Expected almost exclusively in
                           LLM_ONLY. The forced placeholder references
                           (arXiv:0000.00000 "Placeholder Reference N")
                           emitted by llm_only_runner._force_n_references are
                           a *padding artifact*, not a model hallucination,
                           so they are tagged separately as
                           PLACEHOLDER_PADDING and excluded from the genuine
                           fabrication count (recorded bias hazard).

A reference that resolved cleanly (verdict MATCH or MISSING_ID) is NOT a
failure and is classified as ``RESOLVED`` (it never enters the failure
distribution).

This module is pure: it reads a verify.json payload (already produced by
the verify phase) and adds a classification per reference. ZERO network
calls — all the signal it needs (parsed metadata + identification verdict
+ candidate list) was captured during verify.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Optional


# Classification labels.
RESOLVED = "RESOLVED"
METADATA_PARSE_ERROR = "METADATA_PARSE_ERROR"
ID_EXTRACTION_FAIL = "ID_EXTRACTION_FAIL"
DEDUP_COLLISION = "DEDUP_COLLISION"
GENUINELY_FABRICATED = "GENUINELY_FABRICATED"
PLACEHOLDER_PADDING = "PLACEHOLDER_PADDING"

FAILURE_CLASSES = [
    METADATA_PARSE_ERROR,
    ID_EXTRACTION_FAIL,
    DEDUP_COLLISION,
    GENUINELY_FABRICATED,
    PLACEHOLDER_PADDING,
]
ALL_CLASSES = [RESOLVED] + FAILURE_CLASSES


# The forced-padding placeholder signature emitted by
# llm_only_runner._force_n_references. Matching either the sentinel arXiv id
# or the "Placeholder Reference N" title is sufficient and robust.
_PLACEHOLDER_ARXIV_RE = re.compile(r"0000\.00000", re.IGNORECASE)
_PLACEHOLDER_TITLE_RE = re.compile(r"placeholder reference\s*\d+", re.IGNORECASE)


@dataclass
class RefClassification:
    ref_id: Optional[int]
    label: str
    reason: str            # human-readable one-line justification
    verdict: str           # the upstream identification verdict (echoed)

    def to_dict(self) -> dict:
        return asdict(self)


def _is_placeholder_padding(parsed: dict, raw: str) -> bool:
    """Detect the llm_only_runner padding artifact (NOT a hallucination)."""
    arxiv_id = (parsed.get("arxiv_id") or "")
    title = (parsed.get("title") or "")
    if _PLACEHOLDER_ARXIV_RE.search(arxiv_id):
        return True
    if _PLACEHOLDER_TITLE_RE.search(title):
        return True
    if raw and _PLACEHOLDER_TITLE_RE.search(raw):
        return True
    return False


def _metadata_is_malformed(parsed: dict) -> bool:
    """True when the reference line was too malformed to extract a usable
    title. Title is the single field every downstream step needs; if even
    that is missing the reference string itself is broken.
    """
    title = (parsed.get("title") or "").strip()
    return len(title) == 0


def _has_identifier(parsed: dict) -> bool:
    return bool((parsed.get("arxiv_id") or "").strip() or (parsed.get("doi") or "").strip())


def classify_reference(ref_result: dict) -> RefClassification:
    """Classify ONE verify.json per_reference entry into §4.B classes.

    ``ref_result`` is one element of verify.json's ``per_reference`` list,
    i.e. the dict returned by ``_verify_one_reference``:
        {ref_id, raw, parsed:{title,authors,year,arxiv_id,doi},
         existence, resolve_method, candidates, identification:{verdict,...}}

    Decision tree (order matters):
      0. padding placeholder            -> PLACEHOLDER_PADDING
      1. verdict MATCH | MISSING_ID     -> RESOLVED (not a failure)
      2. malformed metadata (no title)  -> METADATA_PARSE_ERROR
      3. verdict MISMATCH               -> DEDUP_COLLISION (id exists, wrong paper)
      4. NOT_FOUND / AMBIGUOUS:
           - had an id/doi              -> GENUINELY_FABRICATED (id given, nothing real)
           - no id, title-only failed   -> ID_EXTRACTION_FAIL (too thin to pin)
    """
    ref_id = ref_result.get("ref_id")
    parsed = ref_result.get("parsed") or {}
    raw = ref_result.get("raw") or ""
    ident = ref_result.get("identification") or {}
    verdict = ident.get("verdict", "NOT_FOUND")

    # 0. padding artifact takes precedence over everything: it is neither a
    #    real citation nor a hallucination, just a forced filler line.
    if _is_placeholder_padding(parsed, raw):
        return RefClassification(
            ref_id=ref_id,
            label=PLACEHOLDER_PADDING,
            reason="forced placeholder reference (llm_only padding artifact)",
            verdict=verdict,
        )

    # 1. clean resolution -> not a failure.
    if verdict in ("MATCH", "MISSING_ID"):
        return RefClassification(
            ref_id=ref_id,
            label=RESOLVED,
            reason=f"resolved cleanly (verdict={verdict})",
            verdict=verdict,
        )

    # 2. the reference string itself is malformed (no extractable title).
    if _metadata_is_malformed(parsed):
        return RefClassification(
            ref_id=ref_id,
            label=METADATA_PARSE_ERROR,
            reason="no extractable title from the reference line",
            verdict=verdict,
        )

    # 3. id was given and the paper EXISTS but resolves to a different paper:
    #    in SurveyFlow this is a dedup collision, not a hallucination.
    if verdict == "MISMATCH":
        return RefClassification(
            ref_id=ref_id,
            label=DEDUP_COLLISION,
            reason="identifier resolves to a different paper than the title (id/title mismatch)",
            verdict=verdict,
        )

    # 4. NOT_FOUND or AMBIGUOUS: split by whether an identifier was present.
    if _has_identifier(parsed):
        # an arXiv id / DOI was given, yet nothing real matches it anywhere:
        # the generator most likely invented the identifier.
        return RefClassification(
            ref_id=ref_id,
            label=GENUINELY_FABRICATED,
            reason=f"identifier present but no real candidate found (verdict={verdict})",
            verdict=verdict,
        )
    # title-only and the lookup could not pin an unambiguous paper.
    return RefClassification(
        ref_id=ref_id,
        label=ID_EXTRACTION_FAIL,
        reason=f"title-only reference could not be pinned to one paper (verdict={verdict})",
        verdict=verdict,
    )


def classify_verify_payload(verify_data: dict) -> dict:
    """Classify every reference in a verify.json payload and aggregate.

    Returns a JSON-serializable dict:
      {
        "system": ..., "topic_id": ..., "seed": ...,
        "n_references": int,
        "per_reference": [RefClassification.to_dict(), ...],
        "distribution": {label: count for label in ALL_CLASSES},
        "n_resolved": int,
        "n_failures": int,                 # excludes RESOLVED and PLACEHOLDER_PADDING
        "n_genuinely_fabricated": int,     # the headline LLM_ONLY signal
      }

    Counts are reported COUNTS-FIRST per §5 (no premature percentages).
    PLACEHOLDER_PADDING is tracked but excluded from both the resolved and
    the failure tallies — it is an artifact, not a datum about the model.
    """
    refs = verify_data.get("per_reference", [])
    classifications = [classify_reference(r) for r in refs]

    distribution = {label: 0 for label in ALL_CLASSES}
    for c in classifications:
        distribution[c.label] = distribution.get(c.label, 0) + 1

    n_resolved = distribution[RESOLVED]
    n_padding = distribution[PLACEHOLDER_PADDING]
    # genuine failures: everything that is neither cleanly resolved nor a
    # forced padding artifact.
    n_failures = sum(
        distribution[label]
        for label in (METADATA_PARSE_ERROR, ID_EXTRACTION_FAIL,
                      DEDUP_COLLISION, GENUINELY_FABRICATED)
    )

    return {
        "system": verify_data.get("system"),
        "topic_id": verify_data.get("topic_id"),
        "seed": verify_data.get("seed"),
        "topic": verify_data.get("topic"),
        "n_references": len(refs),
        "per_reference": [c.to_dict() for c in classifications],
        "distribution": distribution,
        "n_resolved": n_resolved,
        "n_padding": n_padding,
        "n_failures": n_failures,
        "n_genuinely_fabricated": distribution[GENUINELY_FABRICATED],
    }

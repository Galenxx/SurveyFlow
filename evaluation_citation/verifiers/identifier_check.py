"""Identifier-precision checks.

Once a reference has been resolved (by arXiv or S2) to one or more
candidate papers, we need to decide:

  - MATCH         the resolved paper really is the one the survey meant
  - MISMATCH      an ID was provided but resolves to a different paper
  - MISSING_ID    the survey gave a title only, no arXiv ID or DOI,
                  and the title search returned an unambiguous hit
                  (so we attribute the citation as "identified by title")
  - AMBIGUOUS     the title search returned multiple plausible matches
                  and we cannot pick one with confidence

The matching is based on:
  1. title similarity (token Jaccard / SequenceMatcher ratio)
  2. first-author last-name containment
  3. year compatibility (+/- 1 year tolerance for preprints vs venue
     publication year mismatches)

Thresholds come from configs.EXISTENCE_SIM_THRESHOLD and
IDENTIFIER_SIM_THRESHOLD.
"""

from __future__ import annotations
import re
from difflib import SequenceMatcher
from typing import Optional


def _normalize_title(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    # drop punctuation
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def title_similarity(a: str, b: str) -> float:
    """Return 0..1 similarity between two titles.

    Uses the max of:
      - SequenceMatcher ratio on normalized full strings
      - token-Jaccard on the set of lowercase alphanumeric tokens
    """
    na, nb = _normalize_title(a), _normalize_title(b)
    if not na or not nb:
        return 0.0
    sm = SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    jacc = (len(ta & tb) / len(ta | tb)) if (ta | tb) else 0.0
    return max(sm, jacc)


def _last_name(author: str) -> str:
    if not author:
        return ""
    # "Lewis, P." / "Lewis, P. A." / "P. Lewis" -> "lewis"
    if "," in author:
        return author.split(",", 1)[0].strip().lower()
    parts = author.split()
    return parts[-1].strip().lower() if parts else ""


def first_author_last_name(authors_str: Optional[str]) -> str:
    if not authors_str:
        return ""
    # "Kan, L., Yu, Z., & Liu, X." -> "kan"
    first = re.split(r",|&|;", authors_str)[0].strip()
    return _last_name(first)


def year_compatible(survey_year: Optional[str], resolved_year: Optional[str]) -> bool:
    """Year compatibility within +/- 1 year. None / empty always compatible."""
    if not survey_year or not resolved_year:
        return True
    try:
        return abs(int(survey_year) - int(resolved_year)) <= 1
    except (TypeError, ValueError):
        return True


def identify_reference(
    *,
    survey_title: Optional[str],
    survey_authors: Optional[str],
    survey_year: Optional[str],
    survey_arxiv_id: Optional[str],
    survey_doi: Optional[str],
    candidates: list[dict],
    id_sim_threshold: float,
    title_sim_threshold: float,
) -> dict:
    """Decide what the resolved candidates say about this reference.

    Returns:
      {
        "verdict": "MATCH" | "MISMATCH" | "MISSING_ID" | "AMBIGUOUS" | "NOT_FOUND",
        "best_hit": dict | None,
        "title_sim": float,
        "id_match": bool,
      }

    Logic:
      1. If `candidates` is empty -> NOT_FOUND.
      2. If the survey had an arXiv ID or DOI:
           - find the candidate whose id actually matches -> MATCH
           - else -> MISMATCH (it exists, but points to a different paper)
      3. If the survey had no ID at all:
           - rank candidates by title_sim; if top-1 is clearly above the
             threshold and the runner-up is not within 0.05 of it ->
             MISSING_ID (we accept the title identification)
           - if there is a tie near the top -> AMBIGUOUS
           - if top-1 is below the threshold -> AMBIGUOUS
    """
    if not candidates:
        return {
            "verdict": "NOT_FOUND",
            "best_hit": None,
            "title_sim": 0.0,
            "id_match": False,
        }

    # 1. rank by title similarity
    scored = []
    survey_t = survey_title or ""
    for c in candidates:
        sim = title_similarity(survey_t, c.get("title", ""))
        scored.append((sim, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_sim, best_hit = scored[0]
    second_sim = scored[1][0] if len(scored) > 1 else 0.0

    survey_last = first_author_last_name(survey_authors)

    def _author_ok(c: dict) -> bool:
        if not survey_last:
            return True
        for a in c.get("authors") or []:
            if _last_name(a) == survey_last:
                return True
        return False

    def _year_ok(c: dict) -> bool:
        return year_compatible(survey_year, c.get("year"))

    # 2. ID-based identification
    if survey_arxiv_id:
        target = survey_arxiv_id.replace("v1", "").replace("v2", "").replace("v3", "")
        for c in candidates:
            if c.get("arxiv_id") and c["arxiv_id"].replace("v1", "").replace("v2", "").replace("v3", "") == target:
                # ID matches; further confirm by title/year
                if best_sim >= 0.30 and not _year_ok(c):
                    return {
                        "verdict": "MISMATCH",
                        "best_hit": c,
                        "title_sim": best_sim,
                        "id_match": False,
                    }
                return {
                    "verdict": "MATCH" if best_sim >= id_sim_threshold else "MISMATCH",
                    "best_hit": c,
                    "title_sim": best_sim,
                    "id_match": True,
                }
        # ID was given but no candidate shares it
        return {
            "verdict": "MISMATCH",
            "best_hit": best_hit,
            "title_sim": best_sim,
            "id_match": False,
        }

    if survey_doi:
        for c in candidates:
            if c.get("doi") and c["doi"].lower() == survey_doi.lower():
                return {
                    "verdict": "MATCH" if best_sim >= id_sim_threshold else "MISMATCH",
                    "best_hit": c,
                    "title_sim": best_sim,
                    "id_match": True,
                }
        return {
            "verdict": "MISMATCH",
            "best_hit": best_hit,
            "title_sim": best_sim,
            "id_match": False,
        }

    # 3. No ID - title-only identification
    if best_sim >= title_sim_threshold and (best_sim - second_sim) >= 0.05 and _author_ok(best_hit) and _year_ok(best_hit):
        return {
            "verdict": "MISSING_ID",
            "best_hit": best_hit,
            "title_sim": best_sim,
            "id_match": False,
        }
    if best_sim >= title_sim_threshold and (best_sim - second_sim) < 0.05:
        return {
            "verdict": "AMBIGUOUS",
            "best_hit": best_hit,
            "title_sim": best_sim,
            "id_match": False,
        }
    return {
        "verdict": "AMBIGUOUS",
        "best_hit": best_hit,
        "title_sim": best_sim,
        "id_match": False,
    }

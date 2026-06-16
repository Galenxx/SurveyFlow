"""arXiv API verifier.

Given a parsed reference, looks up the paper by arXiv ID (if present)
or by title search. Returns a normalized record with title, authors,
year, abstract - everything the downstream Identifier + Faithfulness
checks need.

API docs: https://info.arxiv.org/help/api/basics.html
"""

from __future__ import annotations
import re
import time
import xml.etree.ElementTree as ET
from typing import Optional
import requests

from evaluation_citation.configs import ARXIV_API_BASE, HTTP_TIMEOUT


_NS = {"a": "http://www.w3.org/2005/Atom"}


def _strip_version(arxiv_id: str) -> str:
    """arXiv IDs can carry a version suffix like 'v2'; the API prefers
    the bare ID for lookups."""
    if arxiv_id and arxiv_id[-2:].startswith("v") and arxiv_id[-2:].isdigit() is False:
        # safe split on the LAST 'v' followed by a digit
        for i in range(len(arxiv_id) - 1, 0, -1):
            if arxiv_id[i] == "v" and arxiv_id[i + 1:].isdigit():
                return arxiv_id[:i]
    return arxiv_id


def _parse_arxiv_response(xml_text: str) -> list[dict]:
    """Parse the Atom XML response from arxiv API into a list of dicts."""
    root = ET.fromstring(xml_text)
    out: list[dict] = []
    for entry in root.findall("a:entry", _NS):
        eid = entry.findtext("a:id", default="", namespaces=_NS)
        # entry id looks like "http://arxiv.org/abs/2405.01580v1"
        arxiv_id = eid.rsplit("/", 1)[-1] if eid else ""
        arxiv_id = _strip_version(arxiv_id)
        title = " ".join((entry.findtext("a:title", default="", namespaces=_NS) or "").split())
        summary = " ".join((entry.findtext("a:summary", default="", namespaces=_NS) or "").split())
        published = entry.findtext("a:published", default="", namespaces=_NS) or ""
        year = published[:4] if published else None
        authors = [
            (a.findtext("a:name", default="", namespaces=_NS) or "").strip()
            for a in entry.findall("a:author", _NS)
        ]
        out.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": summary,
            "year": year,
            "authors": [a for a in authors if a],
            "source": "arxiv",
        })
    return out


def lookup_by_arxiv_id(arxiv_id: str) -> Optional[dict]:
    """Look up a paper by arXiv ID. Returns the first hit or None."""
    if not arxiv_id:
        return None
    bare = _strip_version(arxiv_id)
    params = {"id_list": bare, "max_results": 1}
    try:
        r = requests.get(ARXIV_API_BASE, params=params, timeout=HTTP_TIMEOUT)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    hits = _parse_arxiv_response(r.text)
    return hits[0] if hits else None


def search_by_title(title: str, *, max_results: int = 5) -> list[dict]:
    """Full-text search arXiv by title. Returns up to max_results hits."""
    if not title:
        return []
    # arXiv search treats the whole query as AND across terms; we wrap
    # the title in quotes to bias toward exact matches.
    safe = title.replace('"', "")
    q = f'ti:"{safe}"'
    params = {
        "search_query": q,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        r = requests.get(ARXIV_API_BASE, params=params, timeout=HTTP_TIMEOUT)
    except requests.RequestException:
        return []
    if r.status_code != 200:
        return []
    return _parse_arxiv_response(r.text)


# Generic/structural words that carry little discriminative power in a
# paper title; dropped before building the relaxed query so the search
# keys on the title's distinctive tokens (model names, acronyms, domain
# terms) instead of filler like "framework" / "approach" / "using".
_TITLE_STOPWORDS = frozenset({
    "a", "an", "the", "of", "for", "with", "and", "or", "to", "in", "on",
    "via", "using", "based", "toward", "towards", "from", "into",
    "framework", "approach", "method", "methods", "model", "models",
    "system", "systems", "study", "survey", "review", "analysis", "paper",
    "novel", "new", "efficient", "robust", "deep", "neural", "learning",
    "generation", "two", "stage", "round", "first", "second",
})


def search_by_title_relaxed(title: str, *, max_results: int = 5) -> list[dict]:
    """Relaxed arXiv title search keyed on the title's DISTINCTIVE tokens.

    ``search_by_title`` wraps the whole title in ``ti:"..."`` (exact
    phrase). That fails whenever the citing survey records the title even
    slightly differently from arXiv (e.g. "...Two-stage Framework..." vs the
    real "...Two-Round Refinement..."), which is exactly the case where we
    still need to find the paper to recover its abstract.

    This variant keeps only the title's distinctive tokens (drops generic
    filler and the very structural words that tend to differ between a
    misremembered and the real title), then ANDs the top few as ``ti:``
    terms with relevance ranking. Keying on tokens like an acronym
    ("PET-SQL") or a domain term locates the real record even when the
    surrounding wording is wrong, while ranking by relevance keeps the
    intended paper at or near the top. Callers MUST still gate the result
    with a title-similarity floor — this only widens the recall net.
    """
    if not title:
        return []
    # Tokenize on ANY non-alphanumeric, so hyphens are word BOUNDARIES, not
    # deletions: "PET-SQL" -> [PET, SQL], "Text-to-SQL" -> [Text, to, SQL].
    # arXiv indexes those real sub-tokens; collapsing to "PETSQL" produces a
    # word arXiv has never seen and the AND query returns nothing.
    tokens = re.findall(r"[A-Za-z0-9]+", title)
    distinctive = [
        t for t in tokens
        if len(t) >= 3 and t.lower() not in _TITLE_STOPWORDS
    ]
    # AND only the few most distinctive leading tokens. Too many ANDed terms
    # narrows back toward an exact match (and one wrong word zeroes the
    # result); a small set keyed on acronyms / domain words locates the real
    # record while relevance ranking floats the intended paper to the top.
    terms = distinctive[:4] or tokens[:4]
    if not terms:
        return []
    q = " AND ".join(f"ti:{t}" for t in terms)
    params = {
        "search_query": q,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        r = requests.get(ARXIV_API_BASE, params=params, timeout=HTTP_TIMEOUT)
    except requests.RequestException:
        return []
    if r.status_code != 200:
        return []
    return _parse_arxiv_response(r.text)


# --- public API ---------------------------------------------------------


def resolve_reference(
    *, arxiv_id: Optional[str] = None, title: Optional[str] = None
) -> dict:
    """Resolve a reference via arXiv.

    Returns a dict:
      {
        "found": bool,
        "hits": [ {arxiv_id, title, abstract, year, authors}, ... ],
        "method": "arxiv_id" | "title_search" | "none",
      }

    The first hit (if any) is the one to use; additional hits are kept
    so downstream judges can decide ambiguity.
    """
    if arxiv_id:
        hit = lookup_by_arxiv_id(arxiv_id)
        if hit:
            return {"found": True, "hits": [hit], "method": "arxiv_id"}
    if title:
        # be polite to the API
        time.sleep(0.5)
        hits = search_by_title(title)
        if hits:
            return {"found": True, "hits": hits, "method": "title_search"}
    return {"found": False, "hits": [], "method": "none"}

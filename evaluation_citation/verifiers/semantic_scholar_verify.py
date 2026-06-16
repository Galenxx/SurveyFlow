"""Semantic Scholar API verifier.

Used as a fallback when arXiv lookup fails (e.g. for non-arXiv papers
indexed only in Semantic Scholar or DOI-resolved venues). Returns the
same normalized record shape as arxiv_verify.

API docs: https://api.semanticscholar.org/api-docs/
"""

from __future__ import annotations
from typing import Optional
import requests

from evaluation_citation.configs import S2_API_BASE, S2_API_KEY, HTTP_TIMEOUT


_FIELDS = "title,authors,year,abstract,externalIds,citationCount"


def _headers() -> dict:
    h = {"Accept": "application/json"}
    if S2_API_KEY:
        h["x-api-key"] = S2_API_KEY
    return h


def _normalize(hit: dict) -> dict:
    authors = [a.get("name", "") for a in hit.get("authors", []) if a.get("name")]
    ext = hit.get("externalIds") or {}
    return {
        "arxiv_id": ext.get("ArXiv"),
        "doi": ext.get("DOI"),
        "title": (hit.get("title") or "").strip(),
        "abstract": (hit.get("abstract") or "").strip(),
        "year": str(hit.get("year")) if hit.get("year") is not None else None,
        "authors": authors,
        "source": "semantic_scholar",
        "citation_count": hit.get("citationCount"),
    }


def lookup_by_arxiv_id(arxiv_id: str) -> Optional[dict]:
    """Look up by arXiv ID. S2's `externalIds` field contains ArXiv IDs."""
    if not arxiv_id:
        return None
    url = f"{S2_API_BASE}/ARXIV:{arxiv_id}"
    try:
        r = requests.get(url, params={"fields": _FIELDS}, headers=_headers(), timeout=HTTP_TIMEOUT)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    return _normalize(r.json())


def lookup_by_doi(doi: str) -> Optional[dict]:
    if not doi:
        return None
    url = f"{S2_API_BASE}/DOI:{doi}"
    try:
        r = requests.get(url, params={"fields": _FIELDS}, headers=_headers(), timeout=HTTP_TIMEOUT)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    return _normalize(r.json())


def search_by_title(title: str, *, limit: int = 5) -> list[dict]:
    """Search S2 by free text. Returns normalized hit dicts."""
    if not title:
        return []
    url = f"{S2_API_BASE}/search"
    params = {"query": title, "limit": limit, "fields": _FIELDS}
    try:
        r = requests.get(url, params=params, headers=_headers(), timeout=HTTP_TIMEOUT)
    except requests.RequestException:
        return []
    if r.status_code != 200:
        return []
    data = r.json() or {}
    return [_normalize(h) for h in (data.get("data") or [])]


def resolve_reference(
    *,
    arxiv_id: Optional[str] = None,
    doi: Optional[str] = None,
    title: Optional[str] = None,
) -> dict:
    """Try arXiv ID -> DOI -> title search on S2.

    Returns:
      {
        "found": bool,
        "hits": [ {arxiv_id, doi, title, abstract, year, authors}, ... ],
        "method": "arxiv_id" | "doi" | "title_search" | "none",
      }
    """
    if arxiv_id:
        hit = lookup_by_arxiv_id(arxiv_id)
        if hit:
            return {"found": True, "hits": [hit], "method": "arxiv_id"}
    if doi:
        hit = lookup_by_doi(doi)
        if hit:
            return {"found": True, "hits": [hit], "method": "doi"}
    if title:
        hits = search_by_title(title)
        if hits:
            return {"found": True, "hits": hits, "method": "title_search"}
    return {"found": False, "hits": [], "method": "none"}

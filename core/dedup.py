"""
Utility for deduplicating papers by title similarity.
"""
import re
from difflib import SequenceMatcher


def normalize_title(title: str) -> str:
    """Normalize a title for comparison: lowercase, strip punctuation, collapse spaces."""
    if not title:
        return ""
    t = title.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def title_similarity(title1: str, title2: str) -> float:
    """Return similarity ratio between two titles (0.0 to 1.0)."""
    t1 = normalize_title(title1)
    t2 = normalize_title(title2)
    if not t1 or not t2:
        return 0.0
    return SequenceMatcher(None, t1, t2).ratio()


def dedup_by_title(papers: list[dict], threshold: float = 0.85) -> list[dict]:
    """
    Deduplicate a list of papers by title similarity.
    When papers are considered duplicates (similarity >= threshold),
    keeps the one that appears first in the list (preserves original ordering).

    Papers must have a 'title' key. Unknown-source papers are kept over known duplicates.

    Returns a deduplicated list preserving original ordering of kept papers.
    """
    keep = []
    seen_similar: set[int] = set()

    for i, paper in enumerate(papers):
        if i in seen_similar:
            continue
        kept_title = paper.get("title", "")
        keep.append(paper)
        for j in range(i + 1, len(papers)):
            if j in seen_similar:
                continue
            other_title = papers[j].get("title", "")
            if title_similarity(kept_title, other_title) >= threshold:
                seen_similar.add(j)

    return keep

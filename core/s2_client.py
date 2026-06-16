"""
Semantic Scholar S2 API client for paper search.
Complements arXiv search by providing additional paper coverage.
"""
import os
import re
import time
import requests
from requests.exceptions import ReadTimeout, ConnectionError as RequestsConnectionError
from typing import Optional


class S2Client:
    """Semantic Scholar S2 API client for paper search."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    PAPER_SEARCH_URL = f"{BASE_URL}/paper/search/bulk"

    REQUIRED_FIELDS = (
        "paperId,title,authors,year,url,"
        "externalIds,openAccessPdf,abstract,"
        "citationCount,venue"
    )

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "SEMANTIC_SCHOLAR_API_KEY is required. "
                "Get one at https://www.semanticscholar.org/product/api"
            )
        self._last_request_time: float = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request_time = time.time()

    def _headers(self) -> dict:
        return {"x-api-key": self.api_key}

    def _normalize_s2_query(self, arxiv_query: str) -> str:
        """Convert an arXiv abs:/ti: query to a plain-text S2 query.

        arXiv query: (abs:"large language model" AND abs:"recommendation" AND abs:"llm" ...)
        S2 query:    "large language model" "recommendation" "llm" ...
        """
        query = arxiv_query.strip()
        terms = re.findall(r'(?:abs|ti):"([^"]+)"', query)
        if not terms:
            return arxiv_query
        combined = " ".join(f'"{t}"' for t in terms)
        return combined

    def _build_year_filter(self, year_start: Optional[str], year_end: Optional[str]) -> Optional[str]:
        if not year_start and not year_end:
            return None
        if year_start and not year_end:
            return f"{year_start[:4]}-"
        if year_end and not year_start:
            return f"-{year_end[:4]}"
        return f"{year_start[:4]}-{year_end[:4]}"

    def search(
        self,
        query: str,
        max_results: int = 20,
        year_start: Optional[str] = None,
        year_end: Optional[str] = None,
        offset: int = 0,
    ) -> list[dict]:
        """Search S2 API and return list of paper metadata.

        Args:
            query: Plain-text search query (not arXiv abs: format).
            max_results: Number of results to return (max 1000 per call).
            year_start: Start year filter (e.g. "2020-01-01").
            year_end: End year filter (e.g. "2025-12-31").
            offset: Pagination offset (0, 100, 200, ...).
            Note: The bulk search endpoint does not support an explicit relevance
            sort param. Results are returned in the API's default order
            (broadly relevant first for the given query).

        Returns:
            List of paper dicts with keys: s2_id, title, authors, year, url,
            arxiv_id, open_access_pdf_url, abstract, citation_count, venue.
        """
        params = {
            "query": query,
            "fields": self.REQUIRED_FIELDS,
            "offset": offset,
            "limit": min(max_results, 100),
        }
        year_filter = self._build_year_filter(year_start, year_end)
        if year_filter:
            params["year"] = year_filter

        self._throttle()
        try:
            response = requests.get(
                self.PAPER_SEARCH_URL,
                headers=self._headers(),
                params=params,
                timeout=(10, 60),
            )
        except (ReadTimeout, RequestsConnectionError) as e:
            raise TimeoutError(f"S2 API request timed out or connection failed: {e}")

        if response.status_code == 401:
            raise RuntimeError("S2 API authentication failed. Check your SEMANTIC_SCHOLAR_API_KEY.")
        if response.status_code == 403:
            raise RuntimeError("S2 API access forbidden. Your API key may not have permission for this endpoint.")
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else 60
            raise RuntimeError(f"S2 API error: 429 (Retry-After: {wait}s)")
        if response.status_code != 200:
            raise RuntimeError(f"S2 API error: {response.status_code} - {response.text[:200]}")

        data = response.json()
        papers = []
        for entry in data.get("data", []):
            ext_ids = entry.get("externalIds", {}) or {}
            oa_pdf = entry.get("openAccessPdf") or {}
            authors = entry.get("authors", []) or []
            papers.append({
                "s2_id": entry.get("paperId", ""),
                "title": entry.get("title") or "",
                "authors": [a.get("name", "") for a in authors if a.get("name")],
                "year": entry.get("year"),
                "url": entry.get("url") or "",
                "arxiv_id": ext_ids.get("ArXiv") or "",
                "open_access_pdf_url": oa_pdf.get("url") or "",
                "abstract": entry.get("abstract") or "",
                "citation_count": entry.get("citationCount") or 0,
                "venue": entry.get("venue") or "",
            })
        return papers

    def search_with_retry(
        self,
        query: str,
        max_results: int = 20,
        year_start: Optional[str] = None,
        year_end: Optional[str] = None,
        max_retries: int = 5,
    ) -> list[dict]:
        """Search S2 with exponential backoff retry, respecting Retry-After from 429 responses."""
        all_papers: list[dict] = []
        remaining = max_results
        offset = 0

        while remaining > 0:
            page_size = min(remaining, 100)
            for attempt in range(max_retries):
                try:
                    page = self.search(
                        query=query,
                        max_results=page_size,
                        year_start=year_start,
                        year_end=year_end,
                        offset=offset,
                    )
                    all_papers.extend(page)
                    remaining -= len(page)
                    offset += len(page)
                    if len(page) < page_size or not page:
                        break
                except TimeoutError as e:
                    if attempt < max_retries - 1:
                        wait_time = min(30 * (2 ** attempt), 300)
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(f"S2 search timed out after {max_retries} attempts: {e}")
                except RuntimeError as e:
                    error_msg = str(e)
                    if "429" in error_msg:
                        if attempt < max_retries - 1:
                            m = re.search(r"Retry-After:\s*(\d+)", error_msg)
                            wait_time = int(m.group(1)) if m else min(30 * (2 ** attempt), 300)
                            time.sleep(int(wait_time * 1.5) + 5)
                        else:
                            raise e
                    else:
                        raise
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        raise e

            if remaining > 0 and len(all_papers) == offset:
                break

        return all_papers[:max_results]

    def lookup_papers(self, paper_ids: list[str], fields: Optional[str] = None) -> list[dict]:
        """Batch lookup papers by S2 paper IDs. Up to 500 at a time."""
        if not paper_ids:
            return []
        all_results = []
        for chunk in [paper_ids[i:i + 500] for i in range(0, len(paper_ids), 500)]:
            self._throttle()
            try:
                response = requests.post(
                    f"{self.BASE_URL}/paper/batch",
                    headers=self._headers(),
                    params={"fields": fields or self.REQUIRED_FIELDS},
                    json=chunk,
                    timeout=(10, 60),
                )
            except (ReadTimeout, RequestsConnectionError) as e:
                raise TimeoutError(f"S2 batch lookup failed: {e}")
            if response.status_code != 200:
                raise RuntimeError(f"S2 batch lookup failed: {response.status_code} - {response.text[:200]}")
            data = response.json()
            all_results.extend(data if isinstance(data, list) else [])
        return all_results

    def download_papers(
        self,
        papers: list[dict],
        folder: str,
        rate_limit: float = 1.0,
        max_retries: int = 3,
    ) -> dict[str, str]:
        """Download PDFs for papers that have an openAccessPdf URL (no arxiv_id).

        Each paper dict must have at least one of:
          - open_access_pdf_url: the direct PDF URL from S2 openAccessPdf
          - s2_id: the S2 paper ID

        Papers without an open_access_pdf_url are skipped.

        Returns {s2_id: "success" | "failed: <reason>", ...}
        """
        import re as _re

        os.makedirs(folder, exist_ok=True)
        results: dict[str, str] = {}

        for paper in papers:
            s2_id = paper.get("s2_id", "")
            pdf_url = paper.get("open_access_pdf_url", "")
            title = paper.get("title", "")

            if not pdf_url:
                if s2_id:
                    results[s2_id] = "skipped: no open_access_pdf_url"
                continue

            filename = self._sanitize_filename(title)
            last_error: Exception | None = None

            for attempt in range(max_retries):
                try:
                    self._throttle()
                    response = requests.get(pdf_url, stream=True, timeout=60)
                    if response.status_code == 429:
                        retry_after = response.headers.get("retry-after")
                        wait = int(retry_after) if retry_after and retry_after.isdigit() else 60
                        raise RuntimeError(f"S2 PDF download failed: 429 (Retry-After: {wait}s)")
                    if response.status_code != 200:
                        raise RuntimeError(f"S2 PDF download failed: {response.status_code}")
                    file_path = os.path.join(folder, filename)
                    with open(file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            f.write(chunk)
                    results[s2_id] = "success"
                    break
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    if "429" in err_str and attempt < max_retries - 1:
                        m = _re.search(r"Retry-After:\s*(\d+)", err_str)
                        wait_time = int(m.group(1)) if m else min(30 * (2 ** attempt), 300)
                        time.sleep(int(wait_time * 1.5) + 5)
                    elif attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        results[s2_id] = f"failed: {last_error}"

            time.sleep(rate_limit)

        return results

    def _sanitize_filename(self, filename: str) -> str:
        """Clean filename by removing invalid characters."""
        filename = filename.replace("\n", "").strip()
        filename = re.sub(r"[\/:*?\"<>|]", "_", filename)
        return filename[:100] + ".pdf"

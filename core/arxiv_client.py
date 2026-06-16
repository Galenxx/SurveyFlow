"""
arXiv API client for searching and downloading papers.
Based on download.py patterns.
"""
import os
import re
import time
import requests
import xml.etree.ElementTree as ET
import urllib.parse
from requests.exceptions import ReadTimeout, ConnectionError as RequestsConnectionError
from typing import Optional
from concurrent.futures import ThreadPoolExecutor


class ArxivClient:
    """arXiv API client for paper search and download."""

    BASE_URL = "https://export.arxiv.org/api/query"
    MIN_REQUEST_INTERVAL = 3.0  # seconds between any two requests (search or download)

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._last_request_time: float = 0.0

    def _throttle(self):
        """Enforce global rate limit across all API calls (search + download)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def sanitize_filename(self, filename: str) -> str:
        """Clean filename by removing invalid characters."""
        filename = filename.replace("\n", "").strip()
        filename = re.sub(r"[\/:*?\"<>|]", "_", filename)
        return filename[:100] + ".pdf"

    def search(
        self,
        query: str,
        max_results: int = 20,
        start: int = 0,
        sort_by: str = "relevance",
    ) -> list[dict]:
        """Search arXiv API and return list of paper metadata."""
        # arXiv enforces ~1500 char limit on query
        if len(query) > 1400:
            query = query[:1400]
        # Strip leading AND/OR which makes arXiv return 400
        query = re.sub(r"^\s*(AND|OR)\s+", "", query, flags=re.IGNORECASE)
        encoded_query = urllib.parse.quote_plus(query)
        url = (
            f"{self.BASE_URL}?search_query={encoded_query}"
            f"&start={start}&max_results={max_results}&sortBy={sort_by}"
        )
        self._throttle()
        try:
            response = requests.get(url, timeout=(10, 60))
        except (ReadTimeout, RequestsConnectionError) as e:
            raise TimeoutError(f"arXiv API request timed out or connection failed: {e}")
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else 60
            raise RuntimeError(f"arXiv API error: 429 (Retry-After: {wait}s)")
        if response.status_code != 200:
            raise RuntimeError(f"arXiv API error: {response.status_code}")

        root = ET.fromstring(response.text)
        papers = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            try:
                title = entry.find("{http://www.w3.org/2005/Atom}title").text
                paper_id = entry.find("{http://www.w3.org/2005/Atom}id").text.split("/")[-1]
                pdf_link = entry.find("{http://www.w3.org/2005/Atom}id").text.replace("abs", "pdf")
                authors = [
                    a.find("{http://www.w3.org/2005/Atom}name").text
                    for a in entry.findall("{http://www.w3.org/2005/Atom}author")
                ]
                summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
                papers.append({
                    "arxiv_id": paper_id,
                    "title": title.strip().replace("\n", " "),
                    "pdf_link": pdf_link,
                    "authors": authors,
                    "summary": summary.strip() if summary else "",
                })
            except Exception:
                continue
        return papers

    def search_with_retry(
        self,
        query: str,
        max_results: int = 20,
        max_retries: int = 5,
    ) -> list[dict]:
        """Search with exponential backoff retry, respecting Retry-After from 429 responses."""
        for attempt in range(max_retries):
            try:
                return self.search(query, max_results=max_results)
            except TimeoutError as e:
                if attempt < max_retries - 1:
                    wait_time = min(30 * (2 ** attempt), 300)
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"arXiv search timed out after {max_retries} attempts: {e}")
            except RuntimeError as e:
                error_msg = str(e)
                if "429" in error_msg:
                    if attempt < max_retries - 1:
                        m = re.search(r"Retry-After:\s*(\d+)", error_msg)
                        if m:
                            try:
                                server_wait = int(m.group(1))
                                wait_time = int(server_wait * 1.5) + 5
                            except ValueError:
                                wait_time = 30 * (2 ** attempt)
                        else:
                            wait_time = 30 * (2 ** attempt)
                        time.sleep(wait_time)
                    else:
                        raise e
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise e
        return []

    def download_pdf(
        self,
        pdf_url: str,
        folder: str,
        filename: str,
        timeout: int = 60,
    ) -> str:
        """Download a PDF file to the specified folder."""
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, filename)
        self._throttle()
        response = requests.get(pdf_url, stream=True, timeout=timeout)
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else 60
            raise RuntimeError(f"Download failed: 429 (Retry-After: {wait}s)")
        if response.status_code != 200:
            raise RuntimeError(f"Download failed: {response.status_code}")
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        return file_path

    def download_papers(
        self,
        papers: list[dict],
        folder: str,
        rate_limit: float = 2.0,
        max_retries: int = 3,
    ) -> dict[str, str]:
        """Download multiple papers. Rate limiting is handled globally by _throttle()."""
        os.makedirs(folder, exist_ok=True)
        results = {}

        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")
            last_error = None
            for attempt in range(max_retries):
                try:
                    filename = self.sanitize_filename(paper["title"])
                    self.download_pdf(paper["pdf_link"], folder, filename)
                    results[arxiv_id] = "success"
                    break
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    if "429" in err_str and attempt < max_retries - 1:
                        m = re.search(r"Retry-After:\s*(\d+)", err_str)
                        if m:
                            try:
                                server_wait = int(m.group(1))
                                wait_time = int(server_wait * 1.5) + 5
                            except ValueError:
                                wait_time = 30 * (2 ** attempt)
                        else:
                            wait_time = 30 * (2 ** attempt)
                        time.sleep(wait_time)
                    elif attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        results[arxiv_id] = f"failed: {last_error}"
            time.sleep(rate_limit)

        return results

    def download_papers_parallel(
        self,
        papers: list[dict],
        folder: str,
        rate_limit: float = 2.0,
        max_retries: int = 3,
    ) -> dict[str, str]:
        """Download papers in parallel. Rate limiting is handled globally by _throttle()."""
        os.makedirs(folder, exist_ok=True)
        results = {}

        def download_one(paper: dict) -> tuple[str, str]:
            arxiv_id = paper.get("arxiv_id", "")
            for attempt in range(max_retries):
                try:
                    filename = self.sanitize_filename(paper["title"])
                    self.download_pdf(paper["pdf_link"], folder, filename)
                    return arxiv_id, "success"
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str and attempt < max_retries - 1:
                        m = re.search(r"Retry-After:\s*(\d+)", err_str)
                        if m:
                            try:
                                server_wait = int(m.group(1))
                                wait_time = int(server_wait * 1.5) + 5
                            except ValueError:
                                wait_time = 30 * (2 ** attempt)
                        else:
                            wait_time = 30 * (2 ** attempt)
                        time.sleep(wait_time)
                    elif attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        return arxiv_id, f"failed: {e}"
            return arxiv_id, "failed: unknown"

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(download_one, p): p for p in papers}
            for future in futures:
                arxiv_id, status = future.result()
                results[arxiv_id] = status
                time.sleep(rate_limit / self.max_workers)

        return results

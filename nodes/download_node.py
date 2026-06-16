"""
Download Node: Searches arXiv and Semantic Scholar S2 in parallel,
deduplicates results, selects top papers, downloads PDFs, converts to JSON/TSV.

Retry logic: if successfully downloaded papers < paper_count, generate new
queries via LLM and retry (up to 5 times). Finally trim to paper_count if
exceeded, or proceed with whatever was downloaded.
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor
from core.state import SurveyState
from core.arxiv_client import ArxivClient
from core.s2_client import S2Client
from core.dedup import dedup_by_title
from core.pdf_processor import PDFProcessor
from core.tsv_manager import TSVManager
from core.llm_client import LLMClient
from core.constants import DEFAULT_ARXIV_PATH, DEFAULT_PAPER_COUNT, survey_subdir
from streaming.manager import get_streaming_manager
from prompts.query_prompts import RETRY_QUERY_SYSTEM_PROMPT, RETRY_QUERY_USER_PROMPT_TEMPLATE


# How many extra papers to target per retry to reduce the chance of still being short.
_PAPERS_PER_RETRY_MULTIPLIER = 1.5


def download_node(state: SurveyState) -> SurveyState:
    """Search arXiv and S2 in parallel, deduplicate, select top papers, download.

    Implements a retry loop: after each download cycle, checks how many papers
    were successfully downloaded. If below paper_count, generates new queries via
    LLM and retries — up to 5 times. On exit, trims to paper_count if more were
    downloaded, or proceeds with whatever is available.
    """
    state.current_node = "download_node"
    state.add_log(f"Download Node: Starting with arXiv query - {state.arxiv_query[:80]}...")
    state.add_log(f"Download Node: Starting with S2 query - {state.s2_query[:80]}...")

    try:
        survey_id = state.survey_id
        paper_count = state.paper_count or DEFAULT_PAPER_COUNT
        # Guard against empty survey_id: ``os.path.join(..., "")``
        # would collapse to DEFAULT_ARXIV_PATH itself and the 168
        # PDFs from this run would mix with files from other runs.
        pdf_dir = survey_subdir(survey_id, DEFAULT_ARXIV_PATH)

        year_start = ""
        if state.year_range and state.year_range[0]:
            y = state.year_range[0]
            year_start = y[:4] if len(y) >= 4 else y

        # ---------- Initial search & download ----------
        initial_arxiv_query = state.arxiv_query
        initial_s2_query = state.s2_query

        state.add_log(
            f"Download Node: Cycle 0 — searching with initial queries "
            f"(arxiv={initial_arxiv_query[:60]}..., s2={initial_s2_query[:60]}...)"
        )

        (
            all_paper_list,
            successfully_downloaded_titles,
            combined_status,
        ) = _do_download_cycle(
            state=state,
            arxiv_query=initial_arxiv_query,
            s2_query=initial_s2_query,
            paper_count=paper_count,
            year_start=year_start,
            pdf_dir=pdf_dir,
            already_downloaded_titles=frozenset(),
            cycle_num=0,
        )

        # ---------- Retry loop ----------
        max_retries = 5
        for retry_num in range(1, max_retries + 1):
            downloaded_count = len(successfully_downloaded_titles)
            state.add_log(
                f"Download Node: After cycle {retry_num - 1}, "
                f"{downloaded_count}/{paper_count} papers successfully downloaded"
            )

            if downloaded_count >= paper_count:
                state.add_log(
                    f"Download Node: Target reached ({downloaded_count} >= {paper_count}), stopping retry loop"
                )
                break

            if retry_num == max_retries:
                state.add_log(
                    f"Download Node: Max retries ({max_retries}) reached, proceeding with {downloaded_count} papers"
                )
                break

            # ---------- Generate new queries via LLM ----------
            new_queries = _generate_retry_queries(
                state=state,
                topic=state.user_input,
                original_arxiv_query=initial_arxiv_query,
                original_s2_query=initial_s2_query,
            )

            if not new_queries["arxiv_query"] and not new_queries["s2_query"]:
                state.add_log("Download Node: LLM did not return valid new queries, stopping retry loop")
                break

            retry_arxiv_query = new_queries.get("arxiv_query", "")
            retry_s2_query = new_queries.get("s2_query", "")

            state.add_log(
                f"Download Node: Cycle {retry_num} — new arXiv query: {retry_arxiv_query[:60]}..."
            )
            state.add_log(
                f"Download Node: Cycle {retry_num} — new S2 query: {retry_s2_query[:60]}..."
            )

            # ---------- Search, deduplicate, download ----------
            (
                cycle_paper_list,
                cycle_success_titles,
                cycle_combined_status,
            ) = _do_download_cycle(
                state=state,
                arxiv_query=retry_arxiv_query,
                s2_query=retry_s2_query,
                paper_count=paper_count,
                year_start=year_start,
                pdf_dir=pdf_dir,
                already_downloaded_titles=frozenset(successfully_downloaded_titles),
                cycle_num=retry_num,
            )

            # Accumulate
            all_paper_list.extend(cycle_paper_list)
            successfully_downloaded_titles.update(cycle_success_titles)
            combined_status.update(cycle_combined_status)

        # ---------- Final selection ----------
        final_downloaded_count = len(successfully_downloaded_titles)
        state.add_log(
            f"Download Node: Final download count: {final_downloaded_count}/{paper_count} papers"
        )

        # Select paper_count papers from the downloaded ones (preserve relevance order)
        if final_downloaded_count > paper_count:
            selected_papers = all_paper_list[:paper_count]
            state.add_log(
                f"Download Node: Downloaded {final_downloaded_count} papers, "
                f"trimming to top {paper_count}"
            )
        else:
            selected_papers = all_paper_list
            if final_downloaded_count < paper_count:
                state.add_log(
                    f"Download Node: Only {final_downloaded_count} papers downloaded "
                    f"(target was {paper_count}), proceeding with available papers"
                )

        state.paper_list = selected_papers
        state.download_status = combined_status
        state.failed_papers = [
            pid for pid, status in combined_status.items()
            if not status.startswith("success") and not status.startswith("skipped")
        ]
        state.add_log(
            f"Download Node: Final — "
            f"{sum(1 for s in combined_status.values() if s.startswith('success'))} succeeded, "
            f"{sum(1 for s in combined_status.values() if s.startswith('skipped'))} skipped, "
            f"{len(state.failed_papers)} failed"
        )

        # ---------- Process PDFs ----------
        pdf_files = []
        if os.path.exists(pdf_dir):
            for f in os.listdir(pdf_dir):
                if f.endswith(".pdf"):
                    pdf_files.append(os.path.join(pdf_dir, f))

        if pdf_files:
            state.add_log(f"Download Node: Processing {len(pdf_files)} PDFs with pymupdf")
            processor = PDFProcessor(survey_id)
            batch_results = processor.process_batch(pdf_files, max_workers=4)

            success_results = [r for r in batch_results if r["status"] == "success"]
            state.chroma_collections = [r["collection_name"] for r in success_results if r["collection_name"]]
            state.total_chunks = sum(r["num_chunks"] for r in success_results)
            state.embedding_status = {
                r["pdf_file"]: (r["status"] if r["status"] == "success" else f"failed: {r.get('error', 'unknown')}")
                for r in batch_results
            }

            state.add_log(
                f"Download Node: Processed {len(success_results)}/{len(pdf_files)} papers, "
                f"{state.total_chunks} chunks across {len(state.chroma_collections)} collections"
            )
        else:
            state.add_log("Download Node: No PDF files found to process")

        tsv_mgr = TSVManager()
        try:
            tsv_path = tsv_mgr.merge_json_to_tsv(survey_id)
            state.tsv_path = tsv_path
            state.add_log(f"Download Node: Created TSV at {tsv_path}")
        except (FileNotFoundError, ValueError) as e:
            state.add_log(f"Download Node: No processed data found - {e}")

        state.progress = 0.2
        state.current_node = "splitter_node"
        state.add_log("Download Node: Complete, transitioning to Splitter Node")

    except Exception as e:
        state.add_log(f"Download Node: ERROR - {str(e)}")
        state.error_message = str(e)
        state.status = "FAILED"

    return state


# -----------------------------------------------------------------------------------------
# Helper: single download cycle (search → dedup → download)
# -----------------------------------------------------------------------------------------

def _do_download_cycle(
    state: SurveyState,
    arxiv_query: str,
    s2_query: str,
    paper_count: int,
    year_start: str,
    pdf_dir: str,
    already_downloaded_titles: frozenset,
    cycle_num: int,
) -> tuple[list[dict], set[str], dict]:
    """
    Execute one complete download cycle:
      1. Search arXiv and S2 in parallel
      2. Merge & deduplicate against already-downloaded papers
      3. Select top candidates
      4. Download PDFs
      5. Return (selected_papers, successfully_downloaded_titles, combined_status)

    `already_downloaded_titles` is used to pre-filter so we never re-select
    papers that are already on disk.
    """
    # ---------- Search ----------
    arxiv_papers, s2_papers = _search_parallel(state, paper_count, year_start, arxiv_query, s2_query)

    state.add_log(
        f"Download Node: Cycle {cycle_num} — found {len(arxiv_papers)} from arXiv, "
        f"{len(s2_papers)} from S2"
    )

    # ---------- Unify ----------
    arxiv_unified = [
        {
            "title": p["title"],
            "pdf_link": p["pdf_link"],
            "arxiv_id": p["arxiv_id"],
            "s2_id": None,
            "open_access_pdf_url": "",
            "authors": p.get("authors", []),
            "source": "arxiv",
            "citation_count": 0,
            "year": None,
        }
        for p in arxiv_papers
    ]

    s2_unified = [
        {
            "title": p["title"],
            "pdf_link": p.get("open_access_pdf_url") or "",
            "arxiv_id": p.get("arxiv_id"),
            "s2_id": p.get("s2_id"),
            "open_access_pdf_url": p.get("open_access_pdf_url") or "",
            "authors": p.get("authors", []),
            "source": "s2",
            "citation_count": p.get("citation_count", 0) or 0,
            "year": p.get("year"),
        }
        for p in s2_papers
    ]

    if cycle_num == 0:
        state.s2_raw_results = s2_papers

    merged = s2_unified + arxiv_unified

    # ---------- Deduplicate against previously downloaded titles ----------
    filtered = [p for p in merged if _title_normalized(p["title"]) not in already_downloaded_titles]
    removed_already = len(merged) - len(filtered)
    if removed_already > 0:
        state.add_log(
            f"Download Node: Cycle {cycle_num} — removed {removed_already} "
            f"already-downloaded papers before dedup"
        )

    deduped = dedup_by_title(filtered, threshold=0.85)
    state.add_log(
        f"Download Node: Cycle {cycle_num} — {len(merged) - len(deduped)} duplicates removed, "
        f"{len(deduped)} unique papers remain"
    )

    # Select up to paper_count * multiplier candidates to account for download failures
    selected = deduped[: int(paper_count * _PAPERS_PER_RETRY_MULTIPLIER)]
    state.add_log(
        f"Download Node: Cycle {cycle_num} — selected top {len(selected)} papers for download"
    )

    # ---------- Prepare download lists ----------
    arxiv_to_download = []
    s2_pdf_only = []

    for p in selected:
        if p.get("arxiv_id"):
            arxiv_to_download.append({
                "title": p["title"],
                "pdf_link": f"https://arxiv.org/pdf/{p['arxiv_id']}.pdf",
                "arxiv_id": p["arxiv_id"],
                "authors": p.get("authors", []),
            })
        elif p.get("open_access_pdf_url"):
            s2_pdf_only.append({
                "title": p["title"],
                "open_access_pdf_url": p["open_access_pdf_url"],
                "s2_id": p.get("s2_id", ""),
                "authors": p.get("authors", []),
            })

    # ---------- Download ----------
    os.makedirs(pdf_dir, exist_ok=True)

    arxiv_client = ArxivClient()
    arxiv_results = {}
    if arxiv_to_download:
        arxiv_results = arxiv_client.download_papers(arxiv_to_download, pdf_dir, rate_limit=2.0)
        state.add_log(
            f"Download Node: Cycle {cycle_num} — arXiv download: "
            f"{sum(1 for v in arxiv_results.values() if v.startswith('success'))}/{len(arxiv_results)} succeeded"
        )

    s2_client = S2Client()
    s2_results = {}
    if s2_pdf_only:
        s2_results = s2_client.download_papers(s2_pdf_only, pdf_dir, rate_limit=1.0)
        state.add_log(
            f"Download Node: Cycle {cycle_num} — S2 PDF download: "
            f"{sum(1 for v in s2_results.values() if v.startswith('success'))}/{len(s2_results)} succeeded"
        )

    # ---------- Build combined status & track successful downloads ----------
    combined_status: dict[str, str] = {}
    success_titles: set[str] = set()

    for arxiv_id, status in arxiv_results.items():
        combined_status[arxiv_id] = status
        if status.startswith("success"):
            for item in arxiv_to_download:
                if item["arxiv_id"] == arxiv_id:
                    success_titles.add(_title_normalized(item["title"]))
                    break

    for s2_id, status in s2_results.items():
        combined_status[f"s2:{s2_id}"] = status
        if status.startswith("success"):
            for item in s2_pdf_only:
                if item["s2_id"] == s2_id:
                    success_titles.add(_title_normalized(item["title"]))
                    break

    return selected, success_titles, combined_status


# -----------------------------------------------------------------------------------------
# Helper: parallel arXiv + S2 search (uses provided queries)
# -----------------------------------------------------------------------------------------

def _search_parallel(
    state: SurveyState,
    paper_count: int,
    year_start: str,
    arxiv_query: str,
    s2_query: str,
) -> tuple[list[dict], list[dict]]:
    arxiv_result: list = []
    s2_result: list = []

    def search_arxiv():
        nonlocal arxiv_result
        try:
            arxiv_result = _search_arxiv(state, paper_count, arxiv_query)
        except Exception as e:
            state.add_log(f"Download Node: arXiv search failed: {e}")
            arxiv_result = []

    def search_s2():
        nonlocal s2_result
        try:
            s2_result = _search_s2(state, paper_count, year_start, s2_query)
        except Exception as e:
            state.add_log(f"Download Node: S2 search failed: {e}")
            s2_result = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_arxiv = executor.submit(search_arxiv)
        f_s2 = executor.submit(search_s2)
        f_arxiv.result()
        f_s2.result()

    return (arxiv_result or [], s2_result or [])


def _search_s2(
    state: SurveyState,
    paper_count: int,
    year_start: str,
    s2_query: str,
) -> list[dict]:
    s2 = S2Client()
    s2_query = s2_query.strip()
    min_acceptable_results = min(5, max(2, paper_count // 3))

    if not s2_query:
        state.add_log("Download Node: No S2 query provided, skipping S2 search")
        return []

    try:
        papers = s2.search_with_retry(
            s2_query,
            max_results=paper_count,
            year_start=year_start,
        )
        state.add_log(f"Download Node: Found {len(papers)} papers from S2")
    except Exception as e:
        state.add_log(f"Download Node: S2 search failed: {e}")
        papers = []

    should_try_generic = len(papers) < min_acceptable_results and bool(state.s2_generic_query)
    if should_try_generic:
        state.add_log(
            f"Download Node: S2 returned too few papers ({len(papers)}), trying generic query"
        )
        try:
            generic_papers = s2.search_with_retry(
                state.s2_generic_query,
                max_results=paper_count,
                year_start=year_start,
            )
            state.add_log(f"Download Node: Found {len(generic_papers)} papers from S2 generic query")
            if len(generic_papers) > len(papers):
                papers = generic_papers
        except Exception as e:
            state.add_log(f"Download Node: S2 generic query failed: {e}")

    return papers


def _search_arxiv(
    state: SurveyState,
    paper_count: int,
    arxiv_query: str,
) -> list[dict]:
    arxiv = ArxivClient()
    arxiv_query = arxiv_query.strip()

    if not arxiv_query or len(arxiv_query) < 5:
        topic_clean = re.sub(r"[^\w\s]", " ", state.user_input).strip()
        topic_words = [w for w in topic_clean.split() if len(w) > 2][:4]
        if not topic_words:
            topic_words = [state.user_input[:20]]
        arxiv_query = " AND ".join(f"abs:{w}" for w in topic_words)
        state.add_log(f"Download Node: Using topic-based arXiv query - {arxiv_query}")

    try:
        papers = arxiv.search_with_retry(arxiv_query, max_results=paper_count)
        state.add_log(f"Download Node: Found {len(papers)} papers from arXiv")
    except Exception as e:
        state.add_log(f"Download Node: arXiv search failed: {e}")
        papers = []

    if not papers and state.generic_arxiv_query:
        try:
            papers = arxiv.search_with_retry(state.generic_arxiv_query, max_results=paper_count)
            state.add_log(f"Download Node: Found {len(papers)} papers from arXiv generic query")
        except Exception as e:
            state.add_log(f"Download Node: arXiv generic query failed: {e}")

    if not papers:
        fallback = f"ti:{state.user_input[:20]}"
        state.add_log(f"Download Node: No papers found, trying title-based arXiv query - {fallback}")
        try:
            papers = arxiv.search_with_retry(fallback, max_results=paper_count)
        except Exception as e:
            state.add_log(f"Download Node: arXiv title query failed: {e}")
            papers = []

    return papers


# -----------------------------------------------------------------------------------------
# Helper: LLM-driven new query generation for retry
# -----------------------------------------------------------------------------------------

def _generate_retry_queries(
    state: SurveyState,
    topic: str,
    original_arxiv_query: str,
    original_s2_query: str,
) -> dict[str, str]:
    """
    Ask the LLM to generate new, different search queries based on the topic
    and the queries that have already been tried. Does NOT reveal already-downloaded
    papers to keep the LLM focused on finding strongly-relevant results.
    """
    llm = LLMClient()
    sm = get_streaming_manager()

    user_prompt = RETRY_QUERY_USER_PROMPT_TEMPLATE.format(
        topic=topic,
        original_arxiv_query=original_arxiv_query or "(none)",
        original_s2_query=original_s2_query or "(none)",
    )

    def on_chunk(token: str):
        sm.emit_llm_token(state.survey_id, "download_node", token)

    try:
        raw_response = llm.generate(
            system_prompt=RETRY_QUERY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=512,
            temperature=0.7,
            model="glm-5.1",
            on_chunk=on_chunk,
        )
    except Exception as e:
        state.add_log(f"Download Node: LLM query generation failed: {e}")
        return {"arxiv_query": "", "s2_query": ""}

    arxiv_q = _extract_query(raw_response, "arxiv_query")
    s2_q = _extract_query(raw_response, "s2_query")

    if not arxiv_q and not s2_q:
        state.add_log(f"Download Node: LLM returned unparseable response: {raw_response[:200]}")

    return {"arxiv_query": arxiv_q, "s2_query": s2_q}


def _extract_query(text: str, key: str) -> str:
    """Extract query value from 'arxiv_query: ...' or 's2_query: ...' lines."""
    pattern = rf"^{re.escape(key)}\s*:\s*(.+)$"
    for line in text.strip().splitlines():
        m = re.match(pattern, line.strip(), re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def _title_normalized(title: str) -> str:
    """Return a lowercase, whitespace-collapsed title for set membership."""
    if not title:
        return ""
    t = title.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

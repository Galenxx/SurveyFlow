"""
Splitter Node: Chunk text, embed with GLM-3, store in ChromaDB.
Based on asg_retriever.py and main.py patterns.
"""
import os
import re
import json
import concurrent.futures
from core.state import SurveyState
from core.pdf_processor import PDFProcessor
from core.chroma_client import ChromaManager, legal_pdf, get_chroma_manager
from core.embedder import get_embedder
from core.constants import DEFAULT_ARXIV_PATH, DEFAULT_MAX_WORKERS, survey_subdir
from streaming.manager import get_streaming_manager


def _log(state: SurveyState, msg: str) -> None:
    """Emit a log to both state.logs and the streaming manager."""
    state.add_log(msg)
    try:
        sm = get_streaming_manager()
        sm.emit_log(state.survey_id, "splitter_node", msg)
    except Exception:
        pass


def splitter_node(state: SurveyState) -> SurveyState:
    """Process papers: re-use PDF processing from download_node if available."""
    state.current_node = "splitter_node"
    sm = get_streaming_manager()
    sm.emit_node_start(state.survey_id, "splitter_node", "Processing and embedding papers")
    _log(state, "Splitter Node: Checking text chunking and embedding state")

    try:
        survey_id = state.survey_id
        # Guard against empty survey_id (see core.constants.survey_subdir)
        pdf_dir = survey_subdir(survey_id, DEFAULT_ARXIV_PATH)

        txt_survey_dir = survey_subdir(survey_id, "./data/txt")
        json_files = []
        if os.path.exists(txt_survey_dir):
            json_files = [f for f in os.listdir(txt_survey_dir) if f.endswith(".json")]

        if json_files:
            _log(state, f"Splitter Node: Reusing {len(json_files)} papers processed by download_node")
            _build_citation_meta(state, pdf_dir)
            state.progress = 0.3
            state.current_node = "gap_node" if state.gap_type.value else "retriever_node"
            return state

        if not os.path.exists(pdf_dir):
            _log(state, f"Splitter Node: PDF directory not found - {pdf_dir}")
            state.progress = 0.3
            state.current_node = "gap_node" if state.gap_type.value else "retriever_node"
            return state

        pdf_files = [
            os.path.join(pdf_dir, f)
            for f in os.listdir(pdf_dir)
            if f.endswith(".pdf")
        ]

        if not pdf_files:
            _log(state, "Splitter Node: No PDF files found")
            state.progress = 0.3
            state.current_node = "gap_node" if state.gap_type.value else "retriever_node"
            return state

        _log(state, f"Splitter Node: Processing {len(pdf_files)} PDFs")

        processor = PDFProcessor(survey_id)
        results = processor.process_batch(pdf_files, max_workers=DEFAULT_MAX_WORKERS)

        chroma_collections = []
        total_chunks = 0
        embedding_status = {}

        for result in results:
            cn = result.get("collection_name", "")
            if cn and result.get("status") == "success":
                chroma_collections.append(cn)
                total_chunks += result.get("num_chunks", 0)
                embedding_status[cn] = "success"
            else:
                error = result.get("error", "unknown")
                cn_key = result.get("pdf_file", "unknown")
                embedding_status[cn_key] = f"failed: {error}"

        state.chroma_collections = chroma_collections
        state.total_chunks = total_chunks
        state.embedding_status = embedding_status

        _log(state, f"Splitter Node: Created {len(chroma_collections)} collections with {total_chunks} total chunks")

        _build_citation_meta(state, pdf_dir)

        state.progress = 0.3
        state.current_node = "gap_node" if state.gap_type.value else "retriever_node"

    except Exception as e:
        _log(state, f"Splitter Node: ERROR - {str(e)}")
        sm.emit_error(state.survey_id, "splitter_node", str(e))
        state.error_message = str(e)
        state.status = "FAILED"

    return state


def _build_citation_meta(state: SurveyState, pdf_dir: str) -> None:
    """Build citation_paper_meta from paper_list, using ChromaDB logs as the bridge.

    Pipeline for each paper in paper_list:
      1. Build {pdf_filename -> paper_metadata} by matching title against PDF files.
      2. For any ChromaDB collection not yet matched, use the ChromaDB log file
         (which contains the actual PDF filename) to find the paper's metadata
         via title-keyword matching against paper_list.
    """
    if not state.paper_list or not os.path.exists(pdf_dir):
        return

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    meta_map: dict[str, dict] = {}
    for paper in state.paper_list:
        title = paper.get("title", "")
        if not title:
            continue
        matched_pdf = _find_matching_pdf(title, pdf_files)
        if not matched_pdf:
            continue
        cn = legal_pdf(matched_pdf)
        if cn in state.chroma_collections:
            matched_cn = legal_pdf(matched_pdf)
            if matched_cn != cn:
                continue
            meta_map[cn] = {
                "title": title,
                "authors": paper.get("authors") or [],
                "year": paper.get("year"),
                "arxiv_id": paper.get("arxiv_id") or "",
                "s2_id": paper.get("s2_id") or "",
            }

    chroma_persist = os.environ.get(
        "CHROMADB_PERSIST_DIRECTORY",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "chroma")
    )
    log_dir = os.path.join(chroma_persist, "logs")
    if os.path.exists(log_dir):
        for cn in state.chroma_collections:
            if cn in meta_map:
                continue
            log_path = os.path.join(log_dir, f"{cn}.json")
            if not os.path.exists(log_path):
                continue
            try:
                with open(log_path, encoding="utf-8") as f:
                    log_data = json.load(f)
                if not log_data:
                    continue
                doc_name = log_data[0].get("metadata", {}).get("doc_name", "")
                if not doc_name:
                    continue
                base_doc = doc_name
                if base_doc.lower().endswith(".pdf"):
                    base_doc = base_doc[:-4]
                matched_pdf = None
                for pdf in pdf_files:
                    pdf_base = pdf
                    if pdf.lower().endswith(".pdf"):
                        pdf_base = pdf[:-4]
                    if pdf_base == base_doc or pdf == doc_name:
                        matched_pdf = pdf
                        break
                if not matched_pdf:
                    matched_pdf = _find_matching_pdf(doc_name, pdf_files)
                if matched_pdf:
                    found = False
                    for paper in state.paper_list:
                        ptitle = paper.get("title", "")
                        if not ptitle:
                            continue
                        if _titles_share_keywords(ptitle, doc_name):
                            meta_map[cn] = {
                                "title": ptitle,
                                "authors": paper.get("authors") or [],
                                "year": paper.get("year"),
                                "arxiv_id": paper.get("arxiv_id") or "",
                                "s2_id": paper.get("s2_id") or "",
                            }
                            found = True
                            break
                    if not found:
                        cleaned = _pdf_filename_to_title(doc_name)
                        if cleaned:
                            meta_map[cn] = {
                                "title": cleaned,
                                "authors": [],
                                "year": None,
                                "arxiv_id": "",
                                "s2_id": "",
                            }
            except Exception:
                pass

    for cn, entry in meta_map.items():
        if entry.get("year") is None and entry.get("arxiv_id"):
            arxiv_id = entry["arxiv_id"].strip().rstrip("vV")
            m = re.match(r"^(\d{2})(\d{2})\.(\d+)", arxiv_id)
            if m:
                yy, mm, _ = m.groups()
                year = int(yy) + 2000 if int(yy) < 90 else int(yy) + 1900
                if 2015 <= year <= 2026:
                    entry["year"] = year

    state.citation_paper_meta = meta_map
    _log(state, f"Splitter Node: Built citation_paper_meta for {len(meta_map)} papers")


def _titles_share_keywords(title1: str, title2: str, threshold: int = 3) -> bool:
    stop = {"a", "an", "the", "of", "in", "for", "on", "with", "and", "or",
            "to", "by", "based", "using", "through", "from", "via"}

    def _normalize(s: str) -> set[str]:
        s = s.lower()
        s = s.replace("'", " ").replace("-", " ").replace("_", " ")
        s = s.replace(".", " ").replace(",", " ").replace(":", " ")
        s = s.replace("llms", "llm").replace("llm s", "llm")
        words = s.split()
        return set(w for w in words if len(w) > 2 and w not in stop)
    key1 = _normalize(title1)
    key2 = _normalize(title2)
    overlap = key1 & key2
    return len(overlap) >= threshold


def _pdf_filename_to_title(doc_name: str) -> str:
    if doc_name.lower().endswith(".pdf"):
        doc_name = doc_name[:-4]
    title = doc_name.replace("_", " ")
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _find_matching_pdf(title: str, pdf_files: list[str]) -> str | None:
    filename = title.replace("\n", "").strip()
    filename = re.sub(r"[\/:*?\"<>|]", "_", filename)
    expected_pdf = filename[:100] + ".pdf"

    for pdf in pdf_files:
        if pdf == expected_pdf:
            return pdf

    stop = {"a", "an", "the", "of", "in", "for", "on", "with", "and", "or", "to", "by",
            "based", "using", "through", "from", "via", "using"}
    title_words = set(title.lower().split())
    title_words -= stop
    title_key = {w for w in title_words if len(w) > 3}

    best_match = None
    best_score = 0

    for pdf in pdf_files:
        pdf_words = set(pdf.lower().replace(".pdf", "").replace("_", " ").split())
        pdf_key = {w for w in pdf_words if len(w) > 3}
        overlap = title_key & pdf_key
        if len(overlap) >= 3 and len(overlap) > best_score:
            best_score = len(overlap)
            best_match = pdf

    return best_match

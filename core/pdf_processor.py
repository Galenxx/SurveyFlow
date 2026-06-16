"""
PDF processing pipeline: PDF→text→chunk→embed→ChromaDB.
Uses pymupdf for text extraction (no external CLI required).
"""
import os
import json
import concurrent.futures
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .mineru_client import MinerUClient
from .embedder import get_embedder
from .chroma_client import ChromaManager, legal_pdf, get_chroma_manager


class PDFProcessor:
    """Pipeline for processing PDFs: extract text → split → embed → store."""

    def __init__(self, survey_id: str, embedder=None):
        if not survey_id or not survey_id.strip():
            # Without a survey_id, ``os.path.join("./data/txt", "")``
            # collapses to ``./data/txt`` and the per-run papers end up
            # flattening into the txt root, mixed with files from other
            # runs. Earlier (pre-fix) versions of the GQE benchmark
            # triggered this and produced 168 orphan JSONs. Fail fast
            # here so the caller sees a clear, actionable error instead
            # of a confusing mess of files two hours later.
            raise ValueError(
                "PDFProcessor requires a non-empty survey_id. The GQE "
                "benchmark's _run_fresh() sets it via uuid4().hex; the "
                "API/CLI entry points do the same. If you are calling "
                "PDFProcessor directly, pass a UUID."
            )
        self.survey_id = survey_id
        self.embedder = embedder or get_embedder()
        self.mineru = MinerUClient()
        self.chroma = get_chroma_manager()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=30,
            length_function=len,
            is_separator_regex=False,
        )
        # Ensure txt output dir exists
        self.txt_dir = os.path.join("./data/txt", survey_id)
        os.makedirs(self.txt_dir, exist_ok=True)

    def _save_extracted_json(self, pdf_file: str, extracted: dict) -> str:
        """Save extracted paper data to txt/{survey_id}/{paper_name}.json."""
        base_name = os.path.splitext(os.path.basename(pdf_file))[0]
        title_clean = base_name.strip()
        for char in ['<', '>', ':', '"', "/", "\\", "|", "?", "*", "_"]:
            title_clean = title_clean.replace(char, " ")
        title_clean = title_clean.strip() or f"paper_{self.survey_id}"

        json_path = os.path.join(self.txt_dir, f"{title_clean}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(extracted, f, ensure_ascii=False, indent=4)
        return json_path

    def process_paper(self, pdf_file: str) -> dict:
        """Process a single PDF: extract text, split, embed, store. Returns result dict."""
        result = {
            "pdf_file": pdf_file,
            "collection_name": "",
            "num_chunks": 0,
            "status": "pending",
            "error": "",
        }
        try:
            base_name = os.path.splitext(os.path.basename(pdf_file))[0]
            title_clean = base_name.strip()
            for char in ['<', '>', ':', '"', "/", "\\", "|", "?", "*", "_"]:
                title_clean = title_clean.replace(char, " ")

            collection_name = legal_pdf(title_clean)
            if not collection_name:
                collection_name = legal_pdf(pdf_file)

            # Extract text using pymupdf
            extracted = self.mineru.extract_information_from_pdf(pdf_file)
            full_text = (
                extracted.get("abstract", "") + "\n" +
                extracted.get("introduction", "") + "\n" +
                extracted.get("main_content", "")
            )

            if not full_text or full_text.strip() == "" or len(full_text.strip()) < 100:
                result["status"] = "failed"
                result["error"] = "Empty or too-short text extracted from PDF"
                return result

            # Save JSON for TSV generation
            self._save_extracted_json(pdf_file, extracted)

            # Split text into chunks
            chunks = self.text_splitter.split_text(full_text)
            chunks = [c.replace("\n", " ") for c in chunks]
            if not chunks:
                result["status"] = "failed"
                result["error"] = "Text split produced no chunks"
                return result

            result["num_chunks"] = len(chunks)

            # Embed chunks
            embeddings = self.embedder.embed_documents(chunks)

            # Store in ChromaDB
            metadatas = [{"doc_name": base_name, "chunk_index": i} for i in range(len(chunks))]
            self.chroma.add_documents(
                collection_name=collection_name,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            result["collection_name"] = collection_name
            result["status"] = "success"
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)

        return result

    def process_batch(
        self,
        pdf_files: list[str],
        max_workers: int = 4,
    ) -> list[dict]:
        """Process multiple PDFs in parallel."""
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_paper, pf): pf for pf in pdf_files}
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        return results

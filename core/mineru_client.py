"""
PDF text extraction using pymupdf (PyMuPDF).
Replaces the MinerU CLI dependency with a pure-Python solution.
"""
import os
import re
import json
import pymupdf
from typing import Optional


class MinerUClient:
    """PDF-to-text extractor using pymupdf. No external CLI required."""

    def __init__(self, output_dir: str = "./data/md"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _extract_text_from_pdf(self, pdf_file: str) -> str:
        """Extract all text from a PDF file using pymupdf."""
        text_parts = []
        try:
            doc = pymupdf.open(pdf_file)
            for page in doc:
                page_text = page.get_text("text")
                if page_text:
                    text_parts.append(page_text)
            doc.close()
        except Exception as e:
            raise RuntimeError(f"pymupdf failed to read {pdf_file}: {e}")
        return "\n".join(text_parts)

    def _clean_text(self, text: str) -> str:
        """Remove excessive whitespace and normalize line breaks."""
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            line = line.strip()
            if line:
                cleaned.append(line)
        return "\n\n".join(cleaned)

    def _extract_sections(self, full_text: str) -> dict:
        """Parse extracted text into structured sections."""
        # Try to split by common academic section headers
        section_patterns = [
            r"(?i)^\s*(abstract)\s*$",
            r"(?i)^\s*(introduction)\s*$",
            r"(?i)^\s*(references)\s*$",
        ]

        sections = {}
        current_section = "preamble"
        current_content = []

        lines = full_text.split("\n")
        for line in lines:
            matched = False
            for pattern in section_patterns:
                if re.match(pattern, line.strip()):
                    if current_content:
                        sections[current_section] = "\n".join(current_content).strip()
                    current_section = re.sub(pattern, r"\1", line.strip().lower())
                    current_content = []
                    matched = True
                    break

            if not matched:
                current_content.append(line)

        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def convert_pdf_to_md(self, pdf_file: str, method: str = "auto") -> Optional[str]:
        """Convert PDF to text. Returns path to extracted text file or None on failure.

        Note: Unlike MinerU (which outputs .md), this outputs .txt to data/txt/{survey_id}/
        The actual saving to txt/ is done by process_md_file().
        This method is kept for API compatibility with PDFProcessor.process_paper().
        """
        try:
            text = self._extract_text_from_pdf(pdf_file)
            cleaned = self._clean_text(text)
            if not cleaned or len(cleaned) < 100:
                return None
            return cleaned
        except Exception:
            return None

    def extract_information_from_pdf(self, pdf_file: str) -> dict:
        """Extract title, authors, abstract, introduction, main_content from a PDF."""
        full_text = self._extract_text_from_pdf(pdf_file)
        cleaned = self._clean_text(full_text)

        # Title: first non-empty line that looks like a title (not too long, no numbers at start)
        title = "Untitled"
        for line in cleaned.split("\n"):
            line = line.strip()
            if line and len(line) < 200 and len(line) > 3:
                # Skip lines that start with numbers (page numbers, sections)
                if not re.match(r"^\d", line):
                    title = line
                    break

        # Authors: look for common patterns between title and abstract
        authors = "N/A"
        author_patterns = [
            r"(?i)(?:authors?|by|written by)[:\s]+(.+?)(?=\n\n|\n[A-Z]|$)",
            r"(?i)(?:authors?)[:\s]+\[?[^\]]+\]?\s*(.+?)(?=\n\n|\n[A-Z]|$)",
        ]
        lines = cleaned.split("\n")
        for i, line in enumerate(lines):
            for pattern in author_patterns:
                m = re.match(pattern, line)
                if m:
                    authors = m.group(1).strip()
                    break

        # Abstract: look for "abstract" section
        abstract = "N/A"
        abstract_match = re.search(
            r"(?i)(?:^|\n)(?:abstract|summary|摘要)[:\s]*(.+?)(?=\n\n|\n\s*(?:1\.|introduction|I\. |Keywords))",
            cleaned, re.DOTALL
        )
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            if len(abstract) < 20:
                abstract = "N/A"

        # Introduction: look for "introduction" section
        intro_match = re.search(
            r"(?i)(?:^|\n)\s*(?:1\.?\s*)?(?:introduction|I\.?\s*Introduction)[:\s]*(.+?)(?=\n\n\s*(?:2\.?|Related|Preliminaries|Background|Methods|Experiments|Discussion))",
            cleaned, re.DOTALL
        )
        introduction = "N/A"
        if intro_match:
            introduction = intro_match.group(1).strip()

        # Main content: everything between introduction and references
        main_match = re.search(
            r"(?i)(?:introduction)(.+?)(?:\n\n\s*(?:references|bibliography|paper citations|$))",
            cleaned, re.DOTALL
        )
        main_content = "N/A"
        if main_match:
            main_content = main_match.group(1).strip()

        # If main content is too short, use full text minus abstract/intro
        if main_content == "N/A" or len(main_content) < 200:
            if abstract != "N/A":
                parts = cleaned.split(abstract, 1)
                if len(parts) > 1:
                    main_content = parts[1][:10000]
            else:
                main_content = cleaned[min(len(title) + len(authors) + 200, len(cleaned)):]

        return {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "introduction": introduction,
            "main_content": main_content,
        }

    def convert_batch(self, pdf_files: list[str], method: str = "auto", max_workers: Optional[int] = None) -> dict:
        """Convert multiple PDFs. Returns {pdf_path: extracted_text}."""
        results = {}
        for pdf_file in pdf_files:
            try:
                text = self._extract_text_from_pdf(pdf_file)
                cleaned = self._clean_text(text)
                results[pdf_file] = cleaned if cleaned and len(cleaned) > 100 else None
            except Exception:
                results[pdf_file] = None
        return results

    def process_md_file(self, md_file_path: str, survey_id: str) -> dict:
        """Alias for backward compatibility. Use extract_information_from_pdf() instead."""
        return self.extract_information_from_pdf(md_file_path)

    def extract_information_from_md(self, md_text: str) -> dict:
        """Parse structured sections from already-extracted text."""
        sections = self._extract_sections(md_text)
        return {
            "title": sections.get("preamble", "Untitled")[:200],
            "authors": "N/A",
            "abstract": sections.get("abstract", "N/A"),
            "introduction": sections.get("introduction", "N/A"),
            "main_content": sections.get("preamble", "N/A"),
        }

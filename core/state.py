"""
SurveyState: Pydantic model defining the complete state of the survey generation pipeline.
All fields are updated by pipeline nodes and persist across the LangGraph workflow.
"""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class GapType(str, Enum):
    THEORETICAL = "Theoretical Gap"
    METHODOLOGICAL = "Methodological Gap"
    DATA = "Data Gap"
    APPLICATION = "Application Gap"
    EVALUATION = "Evaluation Gap"
    PERSPECTIVE = "Perspective Gap"
    EMPTY = ""


class PipelineStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SurveyState(BaseModel):
    """Complete state for the survey generation pipeline."""

    # === User Input ===
    user_input: str = Field(default="", description="User-provided natural language topic")
    gap_type: GapType = Field(default=GapType.EMPTY, description="User-selected gap type")
    paper_count: int = Field(default=20, description="Number of papers to download")
    year_range: tuple[str, str] = Field(
        default_factory=lambda: ("2020-01-01", ""),
        description="Year range for paper search, default 2020 to present"
    )
    enable_citations: bool = Field(default=True, description="Whether to inject paper citations")

    # === Query Node Output ===
    arxiv_query: str = Field(default="", description="Generated strict arXiv query string")
    generic_arxiv_query: str = Field(default="", description="Generated generic/fallback arXiv query for when strict query finds no papers")
    s2_query: str = Field(default="", description="Generated Semantic Scholar S2 query string (natural language)")
    s2_generic_query: str = Field(default="", description="Generated generic/fallback S2 query for when strict query finds no papers")
    query_building_log: str = Field(default="", description="Query generation process log")

    # === Download Node Output ===
    survey_id: str = Field(default="", description="Unique UUID for this task")
    tsv_path: str = Field(default="", description="Path to merged TSV file")
    paper_list: list[dict] = Field(
        default_factory=list,
        description="Paper metadata list [{title, pdf_link, authors, source, arxiv_id, s2_id}]"
    )
    download_status: dict[str, str] = Field(
        default_factory=dict,
        description="Download status per paper {paper_id: status}"
    )
    s2_raw_results: list[dict] = Field(
        default_factory=list,
        description="Raw S2 search results before dedup and selection"
    )
    failed_papers: list[str] = Field(
        default_factory=list,
        description="List of paper IDs that failed to download"
    )

    # === Splitter Node Output ===
    chroma_collections: list[str] = Field(
        default_factory=list,
        description="All ChromaDB collection names"
    )
    total_chunks: int = Field(default=0, description="Total number of text chunks")
    embedding_status: dict[str, str] = Field(
        default_factory=dict,
        description="Embedding status per paper {collection_name: status}"
    )
    citation_paper_meta: dict[str, dict] = Field(
        default_factory=dict,
        description=(
            "Mapping from ChromaDB collection_name to paper metadata {title, authors, year, arxiv_id}. "
            "Populated by SplitterNode from paper_list, used by WriterNode for APA citation injection."
        )
    )

    # === Gap Node Output ===
    gap_report: str = Field(default="", description="Gap analysis report content")
    gap_chunks: list[str] = Field(
        default_factory=list,
        description="Retrieved relevant text chunks"
    )

    # === Retriever Node Output ===
    retrieval_results: dict[str, str] = Field(
        default_factory=dict,
        description="100-word description per paper {paper_id: description}"
    )
    retrieval_results_enriched: dict[str, dict] = Field(
        default_factory=dict,
        description=(
            "Enriched retrieval results per paper: {paper_id: {description, full_title, authors, year}}. "
            "Injected from citation_paper_meta so that section_node can correctly populate "
            "cluster_info[titles] and cluster_info[descriptions]."
        )
    )
    description_list: dict[str, str] = Field(
        default_factory=dict,
        description="Gap-aware short description per paper {paper_id: description}"
    )
    citation_data_list: dict[str, list[dict]] = Field(
        default_factory=dict,
        description="Retrieved citation chunks per paper {paper_id: [{source, distance, content}]}"
    )
    citation_data_path: str = Field(
        default="",
        description="Path to persisted citation data JSON"
    )
    sentence_patterns: list[str] = Field(
        default_factory=list,
        description="Generated academic sentence patterns used as retrieval queries"
    )
    tsv_with_retrieval: str = Field(
        default="",
        description="TSV path with retrieval_result column"
    )

    # === Classifier Node Output ===
    labels: dict[str, int] = Field(
        default_factory=dict,
        description="Classification label per paper {paper_id: label}"
    )
    num_classes: int = Field(default=0, description="Number of categories decided by LLM")
    class_definitions: dict[str, str] = Field(
        default_factory=dict,
        description="Class definitions {label: description}"
    )
    tsv_with_labels: str = Field(
        default="",
        description="TSV path with label column"
    )

    # === Section Node Output ===
    section_titles: list[str] = Field(
        default_factory=list,
        description="Section titles for each category"
    )
    cluster_info: dict[int, dict] = Field(
        default_factory=dict,
        description="Detailed info per class {label: {name, papers, descriptions}}"
    )

    # === Outline Node Output ===
    outline: list[tuple[int, str]] = Field(
        default_factory=list,
        description="Survey outline [[1, '1 Abstract'], [1, '2 Introduction'], ...]"
    )
    outline_raw: str = Field(default="", description="Raw LLM-generated outline text")

    # === Writer Node Output ===
    survey_text: str = Field(default="", description="Full generated survey paper text")
    abstract: str = Field(default="", description="Abstract section")
    introduction: str = Field(default="", description="Introduction section")
    future_directions: str = Field(default="", description="Future Directions section")
    conclusion: str = Field(default="", description="Conclusion section")
    references: str = Field(default="", description="References section")
    md_file_path: str = Field(default="", description="Path to the saved md file, e.g. ./data/md/{survey_id}/survey.md")

    # === Pipeline Status ===
    status: PipelineStatus = Field(default=PipelineStatus.PENDING)
    current_node: str = Field(default="", description="Currently executing node name")
    progress: float = Field(default=0.0, description="Overall progress 0.0~1.0")
    error_message: str = Field(default="", description="Error message if failed")
    logs: list[str] = Field(
        default_factory=list,
        description="Execution log entries"
    )

    def add_log(self, message: str) -> None:
        """Append a timestamped log entry."""
        import datetime
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"[{ts}] {message}")

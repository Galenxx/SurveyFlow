"""
File and directory management utilities.
"""
import os
import shutil
from pathlib import Path
from typing import Optional


class FileManager:
    """Manage survey data directories and files."""

    def __init__(self, root: str = "./data"):
        self.root = Path(root)
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Create all required directories."""
        dirs = ["pdf", "md", "tsv", "txt", "chroma", "arxiv_downloads", "survey_states"]
        for d in dirs:
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def get_survey_dir(self, survey_id: str) -> Path:
        """Get the base directory for a survey."""
        return self.root / "surveys" / survey_id

    def cleanup_survey(self, survey_id: str) -> None:
        """Remove all data for a survey."""
        survey_dir = self.get_survey_dir(survey_id)
        if survey_dir.exists():
            shutil.rmtree(survey_dir)

    def get_file_path(self, survey_id: str, file_type: str, filename: str) -> Path:
        """Get path for a specific file."""
        return self.root / file_type / survey_id / filename

    def list_files(self, survey_id: str, file_type: str, extension: str = "") -> list[Path]:
        """List files in a survey's directory."""
        dir_path = self.root / file_type / survey_id
        if not dir_path.exists():
            return []
        if extension:
            return sorted(dir_path.glob(f"*{extension}"))
        return sorted(dir_path.iterdir())

    def get_size_mb(self, survey_id: str) -> float:
        """Get total size of all survey data in MB."""
        total = 0
        for d in ["pdf", "md", "txt", "chroma"]:
            dir_path = self.root / d / survey_id
            if dir_path.exists():
                for f in dir_path.rglob("*"):
                    if f.is_file():
                        total += f.stat().st_size
        return total / (1024 * 1024)

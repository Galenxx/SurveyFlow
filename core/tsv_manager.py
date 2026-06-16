"""
TSV management utilities for survey data.
Based on main.py TSV generation patterns.
"""
import os
import csv
import json
import pandas as pd
from typing import Optional


class TSVManager:
    """Manage TSV files for survey data pipeline."""

    def __init__(self, tsv_dir: str = "./data/tsv", txt_dir: str = "./data/txt"):
        self.tsv_dir = tsv_dir
        self.txt_dir = txt_dir
        os.makedirs(tsv_dir, exist_ok=True)

    def merge_json_to_tsv(self, survey_id: str) -> str:
        """Merge all JSON files from txt/{survey_id}/ into a single TSV. Returns TSV path."""
        txt_survey_dir = os.path.join(self.txt_dir, survey_id)
        if not os.path.exists(txt_survey_dir):
            raise FileNotFoundError(f"Directory not found: {txt_survey_dir}")

        json_files = [f for f in os.listdir(txt_survey_dir) if f.endswith(".json")]
        if not json_files:
            raise ValueError(f"No JSON files found in {txt_survey_dir}")

        records = []
        for jf in json_files:
            with open(os.path.join(txt_survey_dir, jf), encoding="utf-8") as f:
                data = json.load(f)
                records.append(data)

        df = pd.DataFrame(records)
        col_map = {
            "title": "reference paper title",
            "authors": "reference paper citation information",
            "abstract": "reference paper abstract",
            "introduction": "reference paper introduction",
            "main_content": "main content",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        for col in ["retrieval_result", "label"]:
            if col not in df.columns:
                df[col] = ""

        tsv_path = os.path.join(self.tsv_dir, f"{survey_id}.tsv")
        df.to_csv(tsv_path, sep="\t", index=False)
        return tsv_path

    def add_column(self, tsv_path: str, column_name: str, values: dict[str, str]) -> None:
        """Add or update a column in TSV. values maps ref_title → column value."""
        df = pd.read_csv(tsv_path, sep="\t", index_col=0)
        df[column_name] = df.index.map(lambda i: values.get(str(i), ""))
        df.to_csv(tsv_path, sep="\t")

    def read_tsv(self, tsv_path: str) -> pd.DataFrame:
        """Read TSV file into DataFrame."""
        return pd.read_csv(tsv_path, sep="\t", index_col=0)

    def write_tsv(self, tsv_path: str, df: pd.DataFrame) -> None:
        """Write DataFrame to TSV file."""
        df.to_csv(tsv_path, sep="\t")

    def update_retrieval_results(
        self,
        tsv_path: str,
        retrieval_results: dict[str, str],
    ) -> str:
        """Update TSV with retrieval_result column. Returns updated path."""
        df = pd.read_csv(tsv_path, sep="\t", index_col=0)
        if "retrieval_result" not in df.columns:
            df["retrieval_result"] = ""
        normalized_results = {
            self._normalize_title(str(title)): value
            for title, value in retrieval_results.items()
        }
        for idx in df.index:
            title = str(idx)
            value = retrieval_results.get(title)
            if value is None:
                value = normalized_results.get(self._normalize_title(title), "")
            if value:
                df.at[idx, "retrieval_result"] = value
        output_path = tsv_path.replace(".tsv", "_with_retrieval.tsv")
        df.to_csv(output_path, sep="\t")
        return output_path

    def update_labels(
        self,
        tsv_path: str,
        labels: dict[str, int],
    ) -> str:
        """Update TSV with label column. Returns updated path."""
        df = pd.read_csv(tsv_path, sep="\t", index_col=0)
        if "label" not in df.columns:
            df["label"] = -1

        matched_count = 0
        for idx in df.index:
            idx_title = str(idx).strip().lower()
            for label_title, label_id in labels.items():
                label_title_clean = label_title.strip().lower()
                if (label_title_clean in idx_title or idx_title in label_title_clean
                        or self._titles_match(label_title_clean, idx_title)):
                    df.at[idx, "label"] = label_id
                    matched_count += 1
                    break

        output_path = tsv_path.replace(".tsv", "_with_labels.tsv")
        df.to_csv(output_path, sep="\t")
        return output_path

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Normalize title for tolerant matching across TSV rows and collection names."""
        cleaned = title.strip().lower()
        cleaned = " ".join(cleaned.replace("_", " ").replace("-", " ").replace(".", " ").split())
        return cleaned

    @staticmethod
    def _titles_match(a: str, b: str) -> bool:
        """Check if two paper titles refer to the same paper via key word overlap."""
        words_a = set(a.split())
        words_b = set(b.split())
        stop = {"the", "a", "an", "of", "in", "for", "on", "with", "and", "or", "to", "by", "based", "using", "through"}
        words_a -= stop
        words_b -= stop
        overlap = words_a & words_b
        return len(overlap) >= max(3, min(len(words_a), len(words_b)) * 0.5)

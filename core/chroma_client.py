"""
ChromaDB client wrapper. Based on asg_retriever.py patterns.
"""
import os
import re
import json
import uuid
import time
import threading
import chromadb
import numpy as np
from typing import Optional


os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Retry configuration for transient ChromaDB HNSW loading errors.
_QUERY_MAX_RETRIES = 3
_QUERY_RETRY_DELAY = 0.5  # seconds

# Global lock to prevent concurrent query operations from corrupting ChromaDB's
# internal segment reader state. ChromaDB 1.5.x has known issues where parallel
# queries to the same collection can cause "Nothing found on disk" errors even
# when HNSW files exist on disk.
_chroma_query_lock = threading.Lock()


def legal_pdf(filename: str) -> str:
    """Normalize a filename into a legal ChromaDB collection name."""
    pdf_index = filename.lower().rfind(".pdf")
    if pdf_index != -1:
        name_before_pdf = filename[:pdf_index]
    else:
        name_before_pdf = filename
    name_before_pdf = name_before_pdf.strip()
    name = re.sub(r"[^a-zA-Z0-9._-]", "", name_before_pdf)
    name = name.lower()
    while ".." in name:
        name = name.replace("..", ".")
    name = name[:63]
    if len(name) < 3:
        name = name.ljust(3, "0")
    if not re.match(r"^[a-z0-9]", name):
        name = "a" + name[1:]
    if not re.match(r"[a-z0-9]$", name):
        name = name[:-1] + "a"
    ip_pattern = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
    if ip_pattern.match(name):
        name = "ip_" + name
    return name


def _derive_collection_name(title: str) -> str:
    """Derive a ChromaDB collection name from a paper title.

    Mirrors the logic in arxiv_client.sanitize_filename() + legal_pdf()
    so we can compute the collection name without touching the filesystem.
    """
    filename = title.replace("\n", "").strip()
    filename = re.sub(r"[\/:*?\"<>|]", "_", filename)
    pdf_name = filename[:100] + ".pdf"
    return legal_pdf(pdf_name)


class ChromaManager:
    """ChromaDB operations manager."""

    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)

    def create_collection(self, collection_name: str):
        """Create or get an existing collection."""
        try:
            self.client.create_collection(name=collection_name)
        except chromadb.db.base.UniqueConstraintError:
            pass
        return collection_name

    def get_or_create_collection(self, collection_name: str):
        """Get or create a collection and return it."""
        return self.client.get_or_create_collection(name=collection_name)

    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> list[str]:
        """Add documents to a collection. Returns list of IDs."""
        collection = self.get_or_create_collection(collection_name)
        ids = [str(uuid.uuid4()) for _ in documents]
        collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )
        self._log_add(collection_name, ids, documents, metadatas)
        return ids

    def query(
        self,
        collection_name: str,
        query_embeddings: list[list[float]],
        n_results: int = 5,
    ) -> dict:
        """Query the collection and return results with metadata.

        Retries on transient HNSW segment loading errors (e.g. "Nothing found on disk").
        If HNSW index is missing/corrupted, falls back to brute-force search.
        All queries are serialized via a global lock to prevent ChromaDB internal
        state corruption from concurrent access.
        """
        last_error = None
        for attempt in range(_QUERY_MAX_RETRIES):
            with _chroma_query_lock:
                try:
                    collection = self.client.get_collection(name=collection_name)
                    raw_result = collection.query(
                        query_embeddings=query_embeddings,
                        n_results=n_results,
                        include=["documents", "metadatas", "distances"],
                    )
                    if not isinstance(raw_result, dict):
                        import sys
                        print(
                            f"[ChromaManager.query] CRITICAL: collection={collection_name}, "
                            f"result type={type(raw_result).__name__}, "
                            f"is_list={isinstance(raw_result, list)}, "
                            f"len={len(raw_result) if isinstance(raw_result, (list, tuple)) else 'N/A'}, "
                            f"file={__file__}, "
                            f"python={sys.version.split()[0]}",
                            file=sys.stderr,
                            flush=True,
                        )
                        if isinstance(raw_result, list):
                            print(
                                f"[ChromaManager.query] First element type: {type(raw_result[0]).__name__ if raw_result else 'empty'}",
                                file=sys.stderr, flush=True,
                            )
                    return raw_result
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    is_transient = (
                        "hnsw" in err_str.lower()
                        or "nothing found on disk" in err_str.lower()
                        or "internal error" in err_str.lower()
                    )
                    if is_transient and attempt < _QUERY_MAX_RETRIES - 1:
                        time.sleep(_QUERY_RETRY_DELAY * (attempt + 1))
                        continue
                    if is_transient:
                        return self._brute_force_query(collection_name, query_embeddings, n_results)
                    raise
        raise last_error

    def _brute_force_query(
        self,
        collection_name: str,
        query_embeddings: list[list[float]],
        n_results: int,
    ) -> dict:
        """Fallback: compute cosine similarity directly from stored embeddings.

        Used when HNSW index is missing or corrupted. Fetches all embeddings
        and ranks by similarity to the query.
        """
        import sys as _sys

        collection = self.client.get_collection(name=collection_name)
        total = collection.count()
        if total == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

        all_data = collection.get(include=["documents", "metadatas", "embeddings"])

        q_emb = np.array(query_embeddings[0])
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        scores = []
        for i, emb in enumerate(all_data.get("embeddings", [])):
            if emb is None:
                continue
            e = np.array(emb)
            norm = np.linalg.norm(e)
            sim = float(np.dot(q_emb, e / norm) if norm > 0 else 0.0)
            scores.append((i, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:n_results]

        docs = [all_data["documents"][i] for i, _ in top]
        metas = [all_data["metadatas"][i] if all_data["metadatas"] else {} for i, _ in top]
        dists = [1.0 - s for _, s in top]
        ids_ = [all_data["ids"][i] for i, _ in top]

        print(
            f"[ChromaManager._brute_force_query] "
            f"collection={collection_name}, total={total}, returned={len(docs)} "
            f"(HNSW unavailable, using brute-force fallback)",
            file=_sys.stderr, flush=True,
        )
        return {
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
            "ids": [ids_],
        }

    def _log_add(
        self, collection_name: str, ids: list[str],
        documents: list[str], metadatas: list[dict]
    ) -> None:
        """Log added documents to a JSON file."""
        log_dir = os.path.join(self.persist_directory, "logs")
        os.makedirs(log_dir, exist_ok=True)
        logpath = os.path.join(log_dir, f"{collection_name}.json")
        logs = []
        try:
            with open(logpath, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            logs = []
        added = [
            {"chunk_id": ids[i], "metadata": metadatas[i], "page_content": documents[i]}
            for i in range(len(documents))
        ]
        logs.extend(added)
        with open(logpath, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)


class RetrieverSingleton:
    """Singleton wrapper for ChromaManager to prevent multiple DB connections."""
    _instance = None
    _manager: Optional[ChromaManager] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_manager(self) -> ChromaManager:
        if self._manager is None:
            persist_dir = os.getenv("CHROMADB_PERSIST_DIRECTORY", "./data/chroma")
            self._manager = ChromaManager(persist_dir)
        return self._manager

    def close(self) -> None:
        if self._manager is not None:
            self._manager = None


def get_chroma_manager() -> ChromaManager:
    """Get the singleton ChromaManager instance."""
    return RetrieverSingleton().get_manager()

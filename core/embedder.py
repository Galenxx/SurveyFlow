"""
GLM Embedding-3 wrapper using zhipuai SDK.
Replaces HuggingFaceEmbeddings with Zhipu AI embedding model.
"""
import os
from typing import Optional


class GLMEmbedder:
    """GLM Embedding-3 wrapper using the new zhipuai SDK."""

    def __init__(self, api_key: Optional[str] = None):
        import zhipuai
        self.api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
        self._client = None

    def _get_client(self):
        import zhipuai
        if self._client is None:
            self._client = zhipuai.ZhipuAI(api_key=self.api_key)
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch embed multiple texts. Automatically splits into batches of 64 (API limit)."""
        client = self._get_client()
        all_embeddings = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(
                model="embedding-3",
                input=batch,
            )
            all_embeddings.extend(item.embedding for item in response.data)
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return self.embed_documents([text])[0]

    def validate(self) -> None:
        """Validate the API key by making a test call."""
        self.embed_query("validation test")


class SentenceTransformerEmbedder:
    """Fallback embedder using sentence-transformers. Used when GLM Embedding is unavailable."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from langchain_huggingface import HuggingFaceEmbeddings
        self._embedder = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = self._embedder.embed_documents(texts)
        import torch
        if isinstance(results, torch.Tensor):
            return results.tolist()
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)

    def validate(self) -> None:
        self.embed_query("validation test")


def get_embedder() -> "GLMEmbedder | SentenceTransformerEmbedder":
    """Get the best available embedder. Try GLM Embedding first, fallback to sentence-transformers.

    The choice is made at import-time, so the calling code (e.g.
    ``writer_node``) only has to deal with one of two embedder
    classes. Note that selecting GLMEmbedder does NOT contact the
    Zhipu API yet - the API call is deferred to the first
    ``embed_query`` / ``embed_documents`` call. So this function
    succeeds as long as the ``zhipuai`` Python package is importable
    and ``ZHIPUAI_API_KEY`` is set in the environment; the network /
    quota / auth state of the key is checked later when the writer
    actually embeds a chunk.
    """
    api_key = os.getenv("ZHIPUAI_API_KEY")
    tried_zhipu = False
    if api_key:
        tried_zhipu = True
        try:
            return GLMEmbedder(api_key)
        except Exception as e:
            # Fall through to the sentence-transformers path. The most
            # common cause is a missing ``zhipuai`` package, but there
            # are other reasons (Python version mismatch, etc.) so we
            # don't try to be clever here.
            pass
    try:
        return SentenceTransformerEmbedder()
    except ImportError:
        if tried_zhipu:
            raise RuntimeError(
                "GLMEmbedder failed to initialise (likely missing the "
                "'zhipuai' package) and the sentence-transformers fallback "
                "is not installed. Either fix the Zhipu embedder or run "
                "`pip install sentence-transformers`."
            ) from None
        raise RuntimeError(
            "No embedder available. Set ZHIPUAI_API_KEY in .env to use the "
            "Zhipu embedder, or `pip install sentence-transformers` to use "
            "the local fallback."
        )
    except Exception as e:
        raise RuntimeError(f"No embedder available: {e}")

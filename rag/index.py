"""Embedding + index storage (embeddings.npy + chunks.json) and BM25 keyword index."""
from __future__ import annotations

import json
import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

import config

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANK_MODEL, max_length=512)


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """Cross-encoder reads question and chunk together -> more precise relevance score."""
    if not texts:
        return []
    return [float(s) for s in reranker().predict([(query, t) for t in texts])]


def embed(texts: list[str]) -> np.ndarray:
    # The SAME model is used for documents and queries (requirement of vector search).
    return embedder().encode(texts, normalize_embeddings=True, show_progress_bar=len(texts) > 32).astype("float32")


def save_index(chunks: list[dict]) -> None:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectors = embed([c["text"] for c in chunks])
    np.save(config.INDEX_DIR / "embeddings.npy", vectors)
    (config.INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")


class Index:
    def __init__(self):
        path = config.INDEX_DIR / "chunks.json"
        if not path.exists():
            raise FileNotFoundError("Index not found - run `python ingest.py` first.")
        self.chunks: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        self.vectors: np.ndarray = np.load(config.INDEX_DIR / "embeddings.npy")
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def dense(self, query: str, n: int) -> list[tuple[int, float]]:
        q = embed([query])[0]
        scores = self.vectors @ q                       # cosine similarity (normalized vectors)
        top = np.argsort(-scores)[:n]
        return [(int(i), float(scores[i])) for i in top]

    def keyword(self, query: str, n: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:n]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

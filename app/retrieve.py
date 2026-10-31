"""The index: chunks + a BM25 index + (lazily) a dense bge-m3 matrix.

Three retrieval modes, all behind one `search()`:
  bm25    lexical baseline
  dense   cosine similarity over bge-m3 embeddings
  hybrid  reciprocal-rank fusion (RRF) of the two rankings
For dense/hybrid, any chunk whose cosine is below `min_score` is dropped, so a
question the guides can't answer comes back with no evidence instead of junk.
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from app import config, llm
from app.bm25 import BM25
from app.chunking import Chunk, chunk_guide

RRF_K = 60


@dataclass
class Hit:
    chunk: Chunk
    score: float  # mode-specific ranking score
    cosine: float | None = None  # dense similarity, when computed


class Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self.bm25 = BM25([c.embed_text for c in chunks])

    @classmethod
    def build(cls, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> "Index":
        chunks = []
        for path in sorted(config.RAW_DIR.glob("*.txt")):
            chunks += chunk_guide(path.stem, path.read_text(encoding="utf-8"), size, overlap)
        return cls(chunks)

    @cached_property
    def matrix(self) -> np.ndarray:
        # Embedded on first dense query (cached on disk by llm.embed), so BM25-only
        # runs and unit tests never need the proxy.
        return llm.embed([c.embed_text for c in self.chunks])

    def search(self, query: str, *, k: int = config.TOP_K, mode: str = config.RETRIEVER,
               min_score: float = 0.0, destination: str | None = None) -> list[Hit]:
        n = len(self.chunks)
        allowed = np.ones(n, dtype=bool)
        if destination:
            allowed = np.array([c.destination == destination for c in self.chunks])

        cos = None
        if mode in ("dense", "hybrid"):
            cos = self.matrix @ llm.embed([query])[0]
            allowed &= cos >= min_score
        if mode == "dense":
            score = cos
        elif mode == "bm25":
            score = np.asarray(self.bm25.scores(query))
            allowed &= score > 0
        elif mode == "hybrid":
            lex = np.asarray(self.bm25.scores(query))
            score = 1 / (RRF_K + _ranks(cos)) + 1 / (RRF_K + _ranks(lex))
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")

        order = [i for i in np.argsort(-score) if allowed[i]][:k]
        return [Hit(self.chunks[i], float(score[i]), None if cos is None else float(cos[i])) for i in order]


def _ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank of each element (1 = highest score)."""
    ranks = np.empty(len(scores))
    ranks[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
    return ranks

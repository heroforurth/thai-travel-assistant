"""The ONE place that talks to a model (class LiteLLM proxy via the OpenAI client).

Embeddings are cached on disk by (model, text) hash, so re-indexing with a new chunk
size or re-running the eval never pays for the same text twice.
"""
import hashlib
import sqlite3
import time
from collections.abc import Iterator
from functools import lru_cache

import numpy as np
from openai import APIConnectionError, APIStatusError, OpenAI

from app import config


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    # Created lazily so tests and BM25-only runs work without a key.
    return OpenAI(timeout=120, max_retries=3)


def chat(messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 512) -> str:
    for attempt in range(3):  # the shared class proxy occasionally 5xx's under load
        try:
            resp = _client().chat.completions.create(
                model=config.CHAT_MODEL, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except (APIConnectionError, APIStatusError):
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def chat_stream(messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 512) -> Iterator[str]:
    stream = _client().chat.completions.create(
        model=config.CHAT_MODEL, messages=messages, temperature=temperature,
        max_tokens=max_tokens, stream=True,
    )
    for chunk in stream:
        if chunk.choices and (token := chunk.choices[0].delta.content):
            yield token


@lru_cache(maxsize=1)
def _cache() -> sqlite3.Connection:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(config.INDEX_DIR / "embed_cache.sqlite", check_same_thread=False)
    db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
    return db


def _key(text: str) -> str:
    return hashlib.sha256(f"{config.EMBED_MODEL}\n{text}".encode()).hexdigest()


def embed(texts: list[str], batch: int = 32) -> np.ndarray:
    """Return L2-normalised vectors, one row per text (dot product == cosine)."""
    db = _cache()
    keys = [_key(t) for t in texts]
    found: dict[str, np.ndarray] = {}
    for start in range(0, len(keys), 500):  # SQLite caps the number of ? params
        part = keys[start:start + 500]
        rows = db.execute(f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})", part)
        found.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
    missing = [i for i, k in enumerate(keys) if k not in found]
    for start in range(0, len(missing), batch):
        idx = missing[start:start + batch]
        resp = _client().embeddings.create(model=config.EMBED_MODEL, input=[texts[i] for i in idx])
        for i, d in zip(idx, resp.data):
            vec = np.asarray(d.embedding, dtype=np.float32)
            found[keys[i]] = vec
            db.execute("INSERT OR REPLACE INTO emb VALUES (?, ?)", (keys[i], vec.tobytes()))
        db.commit()
    mat = np.stack([found[k] for k in keys])
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)

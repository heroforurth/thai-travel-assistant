"""Chunk the corpus and embed every chunk (fills the on-disk embedding cache).

Run:  python -m app.ingest                       # uses CHUNK_SIZE / CHUNK_OVERLAP
      python -m app.ingest --size 400 --overlap 50
"""
import argparse
import collections
import time

from app import config
from app.retrieve import Index


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=config.CHUNK_SIZE)
    p.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    args = p.parse_args()

    index = Index.build(args.size, args.overlap)
    per_dest = collections.Counter(c.destination for c in index.chunks)
    print(f"{len(index.chunks)} chunks (size={args.size}, overlap={args.overlap}) from {len(per_dest)} guides")
    t = time.time()
    print(f"embedded matrix {index.matrix.shape} with {config.EMBED_MODEL} in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()

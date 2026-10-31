"""All configuration comes from the environment (.env). No secrets live in code.

OpenAI() itself reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment, so
this module only holds the app's own knobs.
"""
import os
import pathlib

from dotenv import load_dotenv

load_dotenv()

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR = pathlib.Path(os.getenv("INDEX_DIR", ROOT / "index"))

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemma-4-E4B-it")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

# Defaults chosen by the sweep in eval/eval_retrieval.py (see README / report).
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
RETRIEVER = os.getenv("RETRIEVER", "hybrid")  # dense | bm25 | hybrid
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.60"))  # dense-cosine floor per chunk

MAX_STEPS = int(os.getenv("MAX_STEPS", "5"))

# The allow-list of destinations the search tool may filter on (= files in data/raw).
DESTINATIONS = sorted(p.stem for p in RAW_DIR.glob("*.txt"))

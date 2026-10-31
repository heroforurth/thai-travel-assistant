"""Turn a Wikivoyage guide into section-aware chunks.

Each chunk remembers its destination and top-level section ("Get in", "Eat", ...) so a
citation can say *where* a fact came from, and the search tool can filter on it.
"""
import re
from dataclasses import asdict, dataclass

SECTION_RE = re.compile(r"^== ([^=].*?) ==\s*$", re.M)
SUBHEAD_RE = re.compile(r"^=+\s*(.+?)\s*=+\s*$", re.M)


@dataclass
class Chunk:
    id: str
    destination: str
    section: str
    text: str

    @property
    def embed_text(self) -> str:
        # Prefixing the location lets "how do I get to Pai" match a chunk whose body
        # never repeats the word "Pai".
        return f"{self.destination} — {self.section}: {self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def split_sections(text: str) -> list[tuple[str, str]]:
    parts = SECTION_RE.split(text)
    sections = [("Overview", parts[0])]
    sections += [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
    return [(name, SUBHEAD_RE.sub(r"\1:", body).strip()) for name, body in sections if body.strip()]


def _pieces(paragraph: str, size: int) -> list[str]:
    """Split one paragraph into <= size pieces, preferring sentence boundaries."""
    if len(paragraph) <= size:
        return [paragraph]
    out, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", paragraph):
        while len(sent) > size:  # a single monster "sentence": hard-wrap it
            out.append(sent[:size])
            sent = sent[size:]
        if cur and len(cur) + 1 + len(sent) > size:
            out.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    return out + ([cur] if cur else [])


def chunk_text(body: str, size: int, overlap: int) -> list[str]:
    """Greedily pack paragraphs into ~size-char chunks; each new chunk starts with the
    last `overlap` chars of the previous one so a fact on a boundary survives."""
    units = [p for para in body.split("\n") if para.strip() for p in _pieces(para.strip(), size)]
    chunks, cur = [], ""
    for u in units:
        if cur and len(cur) + 1 + len(u) > size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = tail[tail.find(" ") + 1:] if " " in tail else tail
        cur = f"{cur}\n{u}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def chunk_guide(destination: str, text: str, size: int, overlap: int) -> list[Chunk]:
    chunks = []
    for section, body in split_sections(text):
        for i, piece in enumerate(chunk_text(body, size, overlap)):
            chunks.append(Chunk(f"{slug(destination)}/{slug(section)}/{i}", destination, section, piece))
    return chunks

"""PDF loading (ingest + normalize) and sentence-aware chunking with overlap."""
from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

_SENT = re.compile(r"(?<=[.!?])\s+")


def clean(text: str) -> str:
    text = text.replace("­", "")                 # soft hyphens
    text = re.sub(r"-\n(?=\w)", "", text)             # words split by hyphenation
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> list[dict]:
    """One record per page: {'text', 'source', 'page'}."""
    pages = []
    for i, page in enumerate(PdfReader(str(path)).pages, start=1):
        t = clean(page.extract_text() or "")
        if t:
            pages.append({"text": t, "source": path.name, "page": i})
    return pages


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Pack whole sentences into chunks of ~`size` chars; repeat ~`overlap` chars of the tail."""
    sentences = [s.strip() for s in _SENT.split(text.replace("\n", " ")) if s.strip()]
    chunks, cur = [], []
    for s in sentences:
        if cur and len(" ".join(cur)) + len(s) + 1 > size:
            chunks.append(" ".join(cur))
            tail, n = [], 0
            for prev in reversed(cur):          # carry the last sentences as overlap
                if n + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                n += len(prev) + 1
            cur = tail
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def doc_title(path: Path) -> str:
    """'07_italian_dietary_allergen_guide.pdf' -> 'Italian dietary allergen guide'."""
    return re.sub(r"^\d+[_-]", "", path.stem).replace("_", " ").replace("-", " ").capitalize()


def load_and_chunk(pdf_dir: Path, size: int, overlap: int) -> list[dict]:
    chunks = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        title = doc_title(pdf)
        for page in load_pdf(pdf):
            for j, c in enumerate(chunk_text(page["text"], size, overlap)):
                # contextual chunk header: the document title travels with every chunk
                chunks.append({"text": f"[{title}] {c}", "source": page["source"], "page": page["page"],
                               "kind": "pdf", "part": j})
    return chunks

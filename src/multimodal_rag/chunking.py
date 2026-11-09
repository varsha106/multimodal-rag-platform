"""Paragraph-aware chunking with configurable overlap."""

from __future__ import annotations

import re

from .models import Chunk, Document
from .text import normalize_whitespace, stable_id


class TextChunker:
    """Split documents into retrieval-sized passages.

    The splitter prefers paragraph and sentence boundaries. A character-based
    fallback handles a very long paragraph; overlap is applied at word level
    to preserve local context across chunk boundaries.
    """

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 140) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_document(self, document: Document) -> list[Chunk]:
        pieces = self._split_text(document.text)
        chunks: list[Chunk] = []
        for index, piece in enumerate(pieces):
            metadata = dict(document.metadata)
            page = _page_for_text(piece)
            if page is not None:
                metadata["page"] = page
            chunks.append(
                Chunk(
                    chunk_id=stable_id(document.document_id, str(index), piece),
                    document_id=document.document_id,
                    title=document.title,
                    text=piece,
                    source=document.source,
                    chunk_index=index,
                    metadata=metadata,
                )
            )
        return chunks

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.split_document(document))
        return chunks

    def _split_text(self, text: str) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        units: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                units.append(paragraph)
            else:
                units.extend(self._split_long_paragraph(paragraph))

        chunks: list[str] = []
        current = ""
        for unit in units:
            candidate = f"{current}\n\n{unit}".strip() if current else unit
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = self._overlap_prefix(current) + unit
            if len(current) > self.chunk_size:
                chunks.extend(self._split_long_paragraph(current))
                current = ""
        if current:
            chunks.append(current)
        return [normalize_whitespace(chunk) for chunk in chunks if chunk.strip()]

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            if len(sentence) > self.chunk_size:
                if current:
                    pieces.append(current)
                    current = ""
                pieces.extend(self._hard_split(sentence))
                continue
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                pieces.append(current)
                current = self._overlap_prefix(current) + sentence
        if current:
            pieces.append(current)
        return pieces

    def _hard_split(self, text: str) -> list[str]:
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [text[start : start + self.chunk_size] for start in range(0, len(text), step)]

    def _overlap_prefix(self, text: str) -> str:
        if not text or self.chunk_overlap == 0:
            return ""
        tail = text[-self.chunk_overlap :].strip()
        return f"{tail} " if tail else ""


def _page_for_text(text: str) -> int | None:
    match = re.search(r"\[Page\s+(\d+)\]", text)
    return int(match.group(1)) if match else None

"""Document loading for text, Markdown, and PDF files.

The production architecture this mirrors also receives scanned pages, tables,
and images. This open-source reference implementation extracts the text layer
from PDFs and keeps page/modality metadata on every record, so additional
extractors can be inserted without changing the retrieval interfaces.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .models import Document
from .text import normalize_whitespace, stable_id

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}


def load_document(path: str | Path) -> Document:
    """Load one supported document from disk."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported document type: {file_path.suffix}")

    if suffix == ".pdf":
        text, metadata = _load_pdf(file_path)
    else:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        metadata = {"format": suffix.lstrip("."), "modality": "text"}

    title = _title_from_text(text) or file_path.stem.replace("_", " ").title()
    return Document(
        document_id=stable_id(str(file_path.resolve()), str(file_path.stat().st_size)),
        title=title,
        text=normalize_whitespace(text),
        source=str(file_path),
        metadata=metadata,
    )


def load_directory(path: str | Path) -> list[Document]:
    """Load every supported file below ``path`` in deterministic order."""

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Document directory does not exist: {root}")

    files = sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES
    )
    return [load_document(file_path) for file_path in files]


def documents_from_texts(items: Iterable[tuple[str, str, str]]) -> list[Document]:
    """Build documents from ``(title, source, text)`` tuples.

    This is the entry point used by the API's in-memory document endpoint and
    by tests that should not depend on files on disk.
    """

    documents: list[Document] = []
    for title, source, text in items:
        clean_text = normalize_whitespace(text)
        documents.append(
            Document(
                document_id=stable_id(source, title, clean_text),
                title=title,
                text=clean_text,
                source=source,
                metadata={"format": "inline", "modality": "text"},
            )
        )
    return documents


def _load_pdf(file_path: Path) -> tuple[str, dict[str, object]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
        raise RuntimeError("PDF support requires the pypdf package") from exc

    reader = PdfReader(str(file_path))
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"[Page {page_number}]\n{page_text}")

    metadata: dict[str, object] = {
        "format": "pdf",
        "modality": "text",
        "pages": len(reader.pages),
    }
    return "\n\n".join(pages), metadata


def _title_from_text(text: str) -> str | None:
    for line in text.splitlines():
        clean = line.strip().lstrip("#").strip()
        if clean.startswith("[Page ") and clean.endswith("]"):
            continue
        if 3 <= len(clean) <= 120:
            return clean
    return None

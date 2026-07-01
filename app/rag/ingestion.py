from __future__ import annotations
import re
from html.parser import HTMLParser
from pathlib import Path

import fitz
import markdown
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.embeddings import embedding_client
from app.vector_store.qdrant_client import upsert_vectors

SUPPORTED_FILE_TYPES = {"pdf", "txt", "md", "markdown"}


class DocumentLoadError(ValueError):
    """Raised when a document cannot be loaded or parsed."""


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:

        if data.strip():
            self._parts.append(data.strip())

    def get_text(self) -> str:
        return "\n\n".join(self._parts)


def load_document(file_path: str | Path, file_type: str) -> str:

    path = Path(file_path)
    normalized_type = file_type.lower().lstrip(".")

    if normalized_type not in SUPPORTED_FILE_TYPES:
        raise DocumentLoadError(
            f"Unsupported file type '{file_type}'. Supported types: pdf, txt, md."
        )
    if not path.exists():
        raise DocumentLoadError(f"File does not exist: {path}")

    if normalized_type == "pdf":
        text = _load_pdf(path)
    elif normalized_type in {"md", "markdown"}:
        text = _load_markdown(path)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    cleaned_text = _normalize_text(text)
    if not cleaned_text:
        raise DocumentLoadError("Document contains no readable text.")
    return cleaned_text


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def ingest_document(
    file_path: str | Path,
    file_type: str,
    source_file: str | None = None,
) -> dict:
    path = Path(file_path)
    raw_text = load_document(path, file_type)
    chunks = chunk_text(raw_text)
    if not chunks:
        raise DocumentLoadError("Document could not be split into chunks.")

    vectors = embedding_client.embed_texts(chunks)

    payloads = [
        {
            "source_file": source_file or path.name,
            "chunk_index": index,
            "text": chunk,
        }
        for index, chunk in enumerate(chunks)
    ]

    stored_count = upsert_vectors(vectors=vectors, payloads=payloads)
    return {
        "status": "success",
        "source_file": source_file or path.name,
        "chunks_stored": stored_count,
    }


def _load_pdf(path: Path) -> str:
    try:
        with fitz.open(path) as document:
            return "\n\n".join(page.get_text("text") for page in document)
    except Exception as exc:
        raise DocumentLoadError("Failed to parse PDF document.") from exc


def _load_markdown(path: Path) -> str:
    raw_markdown = path.read_text(encoding="utf-8", errors="ignore")
    html = markdown.markdown(raw_markdown)
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    return extractor.get_text()


def _normalize_text(text: str) -> str:

    text = text.replace("\x00", " ")
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
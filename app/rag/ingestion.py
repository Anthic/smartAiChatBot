from __future__ import annotations
import re
from html.parser import HTMLParser
from pathlib import Path

import fitz
import markdown
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.embeddings import embedding_client
from app.vector_store.qdrant_client import upsert_vectors

import json
from mistralai.client import Mistral
from app.config import settings


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


def generate_document_insights(text: str) -> tuple[str, list[str]]:
    """Use Mistral AI in JSON mode to generate a summary and suggested questions."""
    client = Mistral(api_key=settings.MISTRAL_API_KEY)
    
 
    sample_text = text[:8000]
    
    prompt = f"""
    Analyze the following document content.
    Provide a concise 3-bullet point summary of the document.
    Also suggest 3 specific questions a user might want to ask about this document.
    
    Format your response EXACTLY as a JSON object with two keys:
    "summary": "bullet 1\\nbullet 2\\nbullet 3",
    "suggested_questions": ["question 1", "question 2", "question 3"]
    
    Document Content:
    {sample_text}
    """.strip()
    
    try:
        response = client.chat.complete(
            model=settings.MISTRAL_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = response.choices[0].message.content or "{}"
        insights = json.loads(content)
        return insights.get("summary", ""), insights.get("suggested_questions", [])
    except Exception:
        return "Summary could not be generated automatically.", ["What does this document cover?"]



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

    # summary and suggested questions
    summary, suggested_questions = generate_document_insights(raw_text)

    return {
        "status": "success",
        "source_file": source_file or path.name,
        "chunks_stored": stored_count,
        "summary": summary,
        "suggested_questions": suggested_questions,
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
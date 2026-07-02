from __future__ import annotations

from app.rag.ingestion import chunk_text
from app.rag.retriever import _filter_relevant_results, format_chunks_for_prompt


def test_chunk_text_splitting():
    text = "Hello World. " * 100  # Long text
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 512 for c in chunks)


def test_filter_relevant_results():
    mock_results = [
        {"text": "Good chunk", "score": 0.85, "source_file": "doc1.txt", "chunk_index": 0},
        {"text": "Weak chunk", "score": 0.40, "source_file": "doc1.txt", "chunk_index": 1},
    ]
    # Default threshold is 0.65, so only "Good chunk" should pass
    filtered = _filter_relevant_results(mock_results)
    assert len(filtered) == 1
    assert filtered[0]["text"] == "Good chunk"
    assert filtered[0]["score"] == 0.85


def test_format_chunks_for_prompt():
    mock_chunks = [
        {"text": "Sample text", "score": 0.90, "source_file": "test.txt", "chunk_index": 2}
    ]
    formatted = format_chunks_for_prompt(mock_chunks)
    assert "[Source 1]" in formatted
    assert "File: test.txt" in formatted
    assert "Chunk: 2" in formatted
    assert "Sample text" in formatted

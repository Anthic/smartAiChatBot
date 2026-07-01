from __future__ import annotations

from typing import TypedDict

from app.config import settings
from app.rag.embeddings import embedding_client
from app.vector_store.qdrant_client import search_vectors


class RetrievedChunk(TypedDict) : 
    text : str
    score : str
    source_file : str
    chunk_index : str
    
class RetrievalError(RuntimeError) :
    """Raised when RAG retrieval fails."""

def retrieve(query_text: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """
    Retrieve relevant document chunks from Qdrant.

    Flow:
    1. Clean user query
    2. Convert query into embedding vector
    3. Search Qdrant for similar chunks
    4. Filter weak matches using score threshold

    Returns:
        A list of relevant chunks sorted by similarity score.
    """
    query = query_text.strip()
    if not query : 
        return []
    limit = top_k or settings.RETRIEVAL_TOP_K
    
    try:
        query_vector = embedding_client.embed_query(query)
        results = search_vectors(query_vector=query_vector, top_k=limit)
    except Exception as e:
        raise RetrievalError("Failed to retrieve relevant document chunks.") from e
    
    return _filter_relevant_results(results)


def _filter_relevant_results(results : list[dict]) -> list[RetrievedChunk]:
    relevant_chunks: list[RetrievedChunk] = []
    
    for result in results:
        score = float(result.get("score",0.0))
        
        if score < settings.RETRIEVAL_SCORE_THRESHOLD:
            continue
        relevant_chunks.append(
            {
                "text" : result.get("text", ""),
                "score" : score,
                "source_file" : result.get("source_file", "unknown")
            }
        )

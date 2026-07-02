from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import settings
import uuid
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue


_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:    
    global _client
    if _client is None:

        _client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
    return _client


def create_collection_if_not_exists() -> None:
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]

    if settings.QDRANT_COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(

                size=settings.EMBEDDING_DIM,

                distance=Distance.COSINE,
            ),
        )
        print(f"[Qdrant] Created collection: {settings.QDRANT_COLLECTION_NAME}")
    else:
        print(f"[Qdrant] Collection already exists: {settings.QDRANT_COLLECTION_NAME}")


def upsert_vectors(vectors: list[list[float]], payloads: list[dict]) -> int:
    client = get_qdrant_client()

    points = [
        PointStruct(

            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload,
        )
        for vector, payload in zip(vectors, payloads)
    ]

    client.upsert(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points=points,
    )
    return len(points)


def search_vectors(
    query_vector: list[float],
    top_k: int = 3,
    source_files: list[str] | None = None,
) -> list[dict]:
    client = get_qdrant_client()
    query_filter = None
    if source_files:
        query_filter = Filter(
            should=[
                FieldCondition(
                    key="source_file",
                    match=MatchValue(value=filename)
                )
                for filename in source_files
            ]
        )

    response = client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True, 
    )


    return [
        {
            "text": hit.payload.get("text", ""),
            "score": hit.score,
            "source_file": hit.payload.get("source_file", "unknown"),
            "chunk_index": hit.payload.get("chunk_index", -1),
        }
        for hit in response.points
    ]


def ping_qdrant() -> dict:
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    collection_exists = settings.QDRANT_COLLECTION_NAME in collections

    info = client.get_collection(settings.QDRANT_COLLECTION_NAME) if collection_exists else None


    points_count = 0
    if info:
        points_count = getattr(info, "points_count", None) or getattr(info, "vectors_count", 0) or 0

    return {
        "connected": True,
        "collection": settings.QDRANT_COLLECTION_NAME,
        "collection_exists": collection_exists,
        "vectors_count": points_count,
    }
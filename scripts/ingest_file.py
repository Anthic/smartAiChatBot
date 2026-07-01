from __future__ import annotations

import sys
from pathlib import Path

from app.rag.ingestion import ingest_document
from app.vector_store.qdrant_client import create_collection_if_not_exists, ping_qdrant


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/ingest_file.py <file_path>")
        raise SystemExit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"File not found: {file_path}")
        raise SystemExit(1)

    file_type = file_path.suffix.lower().lstrip(".")

    print("Ensuring Qdrant collection exists...")
    create_collection_if_not_exists()

    print(f"Ingesting file: {file_path.name}")
    result = ingest_document(
        file_path=file_path,
        file_type=file_type,
        source_file=file_path.name,
    )

    print("Ingestion result:")
    print(result)

    print("Qdrant status:")
    print(ping_qdrant())


if __name__ == "__main__":
    main()
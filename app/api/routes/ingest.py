from __future__ import annotations

import tempfile
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.models.ingest import IngestResponse
from app.rag.ingestion import SUPPORTED_FILE_TYPES, DocumentLoadError, ingest_document


router = APIRouter(prefix="/ingest", tags=["RAG Ingestion"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@router.post("", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
    original_name = file.filename or "uploaded_document"
    file_type = Path(original_name).suffix.lower().lstrip(".")

    if file_type not in SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Upload a PDF, TXT, or Markdown file.",
        )

    temp_path = await _save_upload_to_temp(file, file_type)

    try:
        result = await run_in_threadpool(
            ingest_document,
            temp_path,
            file_type,
            original_name,
        )
    except DocumentLoadError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document ingestion failed.",
        ) from exc
    finally:
        temp_path.unlink(missing_ok=True)

    return IngestResponse(**result)


async def _save_upload_to_temp(file: UploadFile, file_type: str) -> Path:
    total_bytes = 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as temp_file:
        temp_path = Path(temp_file.name)

    async with aiofiles.open(temp_path, "wb") as out_file:
        while chunk := await file.read(1024 * 1024):
            total_bytes += len(chunk)

            if total_bytes > MAX_UPLOAD_BYTES:
                temp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="File is too large. Max upload size is 10 MB.",
                )

            await out_file.write(chunk)

    return temp_path
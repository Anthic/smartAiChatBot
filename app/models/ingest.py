from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    status: str = Field(examples=["success"])
    source_file: str
    chunks_stored: int
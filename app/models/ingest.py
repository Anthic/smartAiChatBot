from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    status: str = Field(examples=["success"])
    source_file: str
    chunks_stored: int
    summary: str | None = Field(
        None,
        description="A 3-bullet summary of the document.",
        examples=["Returns are accepted within 30 days.\n- Items must be in original packaging.\n- Return shipping is free."],
    )
    suggested_questions: list[str] | None = Field(
        None,
        description="3 sample questions the user can ask about this document.",
        examples=["What is the return window?", "Is return shipping free?", "Can I return open items?"],
    )

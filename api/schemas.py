"""
FastAPI Pydantic request/response schemas.
"""
from typing import Optional
from pydantic import BaseModel, Field


class SurveyGenerateRequest(BaseModel):
    topic: str = Field(..., description="Research topic in natural language")
    gap_type: Optional[str] = Field(
        default="",
        description="Gap type: Theoretical Gap, Methodological Gap, Data Gap, "
                   "Application Gap, Evaluation Gap, Perspective Gap, or empty"
    )
    paper_count: int = Field(default=20, ge=5, le=100, description="Number of papers to download")
    year_range: Optional[list[str]] = Field(
        default=None,
        description="Year range as [start, end], e.g. ['2020-01-01', '2025-12-31']"
    )
    enable_citations: bool = Field(default=True, description="Whether to inject paper citations")


class SurveyGenerateResponse(BaseModel):
    survey_id: str
    status: str
    message: str


class SurveyStatusResponse(BaseModel):
    survey_id: str
    status: str
    current_node: str
    progress: float
    error_message: str
    logs: list[str]


class SurveyResultResponse(BaseModel):
    survey_id: str
    status: str
    result: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    version: str


class StreamMessage(BaseModel):
    type: str
    content: str

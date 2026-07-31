from enum import Enum

from pydantic import BaseModel, Field


class ResultVerdict(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PARTIAL = "PARTIAL"
    INVALID = "INVALID"


class ResolveRequest(BaseModel):
    claim: str
    success_condition: str
    failure_condition: str
    actual_result: str
    evidence_summary: str


class ResolutionAssessment(BaseModel):
    verdict: ResultVerdict
    ai_reason: str = Field(description="판정 근거를 한국어 2~4문장으로 설명")


class ResolveResponse(BaseModel):
    verdict: ResultVerdict
    ai_reason: str

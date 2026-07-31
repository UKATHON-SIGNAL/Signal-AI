from enum import Enum

from pydantic import BaseModel, Field


class ReviewLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    CAUTION = "CAUTION"


class VerificationStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"


class SourceInput(BaseModel):
    url: str
    title: str | None = None


class VerifyRequest(BaseModel):
    claim: str
    success_condition: str
    failure_condition: str
    evidence_summary: str
    category: str | None = None
    sources: list[SourceInput] = Field(default_factory=list)
    creator_average_score: float | None = None
    creator_evaluated_count: int = 0


class EvidenceAssessment(BaseModel):
    evidence_relevance_level: ReviewLevel
    evidence_relevance_comment: str = Field(
        description="출처가 주장을 얼마나 뒷받침하는지 한국어 2~3문장으로 설명"
    )


class CounterAssessment(BaseModel):
    counterargument_level: ReviewLevel = Field(
        description="반대 논리가 강할수록 CAUTION/LOW, 약할수록 HIGH"
    )
    counterargument_comment: str
    missing_variable_level: ReviewLevel = Field(
        description="판정에 필요한 변수가 잘 갖춰져 있으면 HIGH, 많이 빠졌으면 CAUTION/LOW"
    )
    missing_variable_comment: str


class SynthesisAssessment(BaseModel):
    status: VerificationStatus
    generated_title: str = Field(description="카드 목록에 노출될 15자 내외의 제목")
    generated_summary: str = Field(description="2~3문장의 카드 요약")
    recommended_price_min: int = Field(description="원화 기준 추천 최저 판매가")
    recommended_price_max: int = Field(description="원화 기준 추천 최고 판매가")
    duplication_score: float = Field(description="기존에 널리 알려진 정보와의 중복도, 0~100")
    overall_comment: str


class VerifyResponse(BaseModel):
    status: VerificationStatus
    generated_title: str
    generated_summary: str
    recommended_price_min: int
    recommended_price_max: int
    evidence_relevance_level: ReviewLevel
    evidence_relevance_comment: str
    missing_variable_level: ReviewLevel
    missing_variable_comment: str
    counterargument_level: ReviewLevel
    counterargument_comment: str
    duplication_score: float
    overall_comment: str

from enum import Enum

from pydantic import BaseModel, Field


class TrendDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


class CardSummaryInput(BaseModel):
    claim: str
    category: str | None = None


class BriefingRequest(BaseModel):
    cards: list[CardSummaryInput] = Field(default_factory=list)


class BriefingInsight(BaseModel):
    title: str = Field(description="15자 내외의 트렌드 요약 문구")
    trend: TrendDirection


class BriefingAssessment(BaseModel):
    insights: list[BriefingInsight] = Field(description="정확히 3개")


class BriefingResponse(BaseModel):
    insights: list[BriefingInsight]

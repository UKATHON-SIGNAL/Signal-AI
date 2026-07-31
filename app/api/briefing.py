from fastapi import APIRouter, HTTPException

from app.agents.briefing_agent import run_briefing_agent
from app.schemas.briefing import BriefingRequest, BriefingResponse

router = APIRouter()


@router.post("/api/briefing", response_model=BriefingResponse)
def briefing(request: BriefingRequest) -> BriefingResponse:
    try:
        result = run_briefing_agent([card.model_dump() for card in request.cards])
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return BriefingResponse(insights=result["insights"])

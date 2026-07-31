from fastapi import APIRouter, HTTPException

from app.agents.refinement_agent import run_refinement_agent
from app.schemas.draft_assist import DraftAssistRequest, DraftAssistResponse

router = APIRouter()


@router.post("/api/draft-assist", response_model=DraftAssistResponse)
def draft_assist(request: DraftAssistRequest) -> DraftAssistResponse:
    try:
        result = run_refinement_agent(request.action, request.text, request.category)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return DraftAssistResponse(result=result)

from fastapi import APIRouter, HTTPException

from app.agents.resolution_agent import run_resolution_agent
from app.schemas.resolve import ResolveRequest, ResolveResponse

router = APIRouter()


@router.post("/api/resolve", response_model=ResolveResponse)
def resolve(request: ResolveRequest) -> ResolveResponse:
    try:
        result = run_resolution_agent(
            claim=request.claim,
            success_condition=request.success_condition,
            failure_condition=request.failure_condition,
            actual_result=request.actual_result,
            evidence_summary=request.evidence_summary,
            evaluation_metric=request.evaluation_metric,
            full_hit_threshold=request.full_hit_threshold,
            partial_hit_threshold=request.partial_hit_threshold,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return ResolveResponse(verdict=result["verdict"], ai_reason=result["ai_reason"])

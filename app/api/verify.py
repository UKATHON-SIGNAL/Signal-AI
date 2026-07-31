from fastapi import APIRouter, HTTPException

from app.agents.verify_graph import run_verify_graph
from app.schemas.verify import VerifyRequest, VerifyResponse

router = APIRouter()


@router.post("/api/verify", response_model=VerifyResponse)
def verify(request: VerifyRequest) -> VerifyResponse:
    try:
        result = run_verify_graph(
            claim=request.claim,
            success_condition=request.success_condition,
            failure_condition=request.failure_condition,
            evidence_summary=request.evidence_summary,
            category=request.category,
            sources=[source.model_dump() for source in request.sources],
            creator_average_score=request.creator_average_score,
            creator_evaluated_count=request.creator_evaluated_count,
            creator_source_reliability=request.creator_source_reliability,
            days_until_result=request.days_until_result,
            existing_duplication_score=request.existing_duplication_score,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return VerifyResponse(**result)

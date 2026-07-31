from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from app.config import settings
from app.schemas.resolve import ResolutionAssessment

_SYSTEM_PROMPT = (
    "당신은 정보 카드의 결과를 판정하는 Resolution Agent입니다. "
    "카드의 주장, 성공 조건, 실패 조건과 작성자가 제출한 실제 결과·근거를 비교해서 "
    "주장이 맞았는지 판정하세요.\n"
    "- 성공 조건을 명확히 충족하면 SUCCESS\n"
    "- 실패 조건을 명확히 충족하면 FAILURE\n"
    "- 방향은 맞았지만 조건을 정확히 충족했다고 보기 애매하면 PARTIAL\n"
    "- 제출된 실제 결과·근거가 판정 조건과 무관하거나 판단할 근거가 부족하면 INVALID"
)


class ResolutionState(TypedDict):
    claim: str
    success_condition: str
    failure_condition: str
    actual_result: str
    evidence_summary: str
    verdict: str
    ai_reason: str


def _resolve(state: ResolutionState) -> dict:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not configured")

    human_content = (
        f"주장: {state['claim']}\n"
        f"성공 조건: {state['success_condition']}\n"
        f"실패 조건: {state['failure_condition']}\n\n"
        f"제출된 실제 결과: {state['actual_result']}\n"
        f"제출된 근거: {state['evidence_summary']}"
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.groq_api_key)
    structured_llm = llm.with_structured_output(ResolutionAssessment)
    result = structured_llm.invoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_content)]
    )
    return {"verdict": result.verdict, "ai_reason": result.ai_reason}


def build_resolution_graph():
    graph = StateGraph(ResolutionState)
    graph.add_node("resolve", _resolve)
    graph.set_entry_point("resolve")
    graph.add_edge("resolve", END)
    return graph.compile()


_resolution_graph = build_resolution_graph()


def run_resolution_agent(
    claim: str,
    success_condition: str,
    failure_condition: str,
    actual_result: str,
    evidence_summary: str,
) -> dict:
    return _resolution_graph.invoke(
        {
            "claim": claim,
            "success_condition": success_condition,
            "failure_condition": failure_condition,
            "actual_result": actual_result,
            "evidence_summary": evidence_summary,
            "verdict": "",
            "ai_reason": "",
        }
    )

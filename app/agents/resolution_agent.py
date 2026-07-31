from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from app.config import settings
from app.schemas.resolve import ResolutionAssessment

_SYSTEM_PROMPT = (
    "당신은 정보 카드의 결과를 판정하는 Resolution Agent입니다. "
    "카드 발행 시점에 작성자가 미리 정해둔 성공 조건과 실패 조건 문장을 기준으로, "
    "작성자가 제출한 실제 결과·근거를 대조해서 판정하세요. 새로운 기준을 만들지 말고 "
    "주어진 성공/실패 조건 문장에 담긴 구체적인 수치·기준을 그대로 사용하세요.\n\n"
    "성공 조건과 실패 조건 문장에는 보통 구체적인 임계값(예: '0.25%p 낮을 경우')이 "
    "포함되어 있습니다. 그 임계값을 기준으로 아래 순서대로 판정하세요:\n"
    "1. 실제 결과가 성공 조건의 임계값을 그대로 충족한다 → SUCCESS\n"
    "2. 실패 조건을 충족하지 않고, 성공 조건과 같은 방향으로 임계값의 절반 이상 "
    "움직였지만 정확히 충족하지는 못했다 → PARTIAL\n"
    "3. 실패 조건을 충족하지 않고, 성공 조건과 같은 방향이긴 하지만 임계값의 절반에도 "
    "못 미치게 미미하게 움직였다 → DIRECTION_ONLY\n"
    "4. 실패 조건을 충족하거나 예측한 방향과 반대로 움직였다 → FAILURE\n"
    "5. 실제 결과·근거가 조건과 무관하거나 판단할 근거가 부족하다 → INVALID\n\n"
    "예시: 성공 조건이 '정책금리가 0.25%p 낮을 경우'이고 실제로 0.15%p 낮아졌다면 "
    "임계값의 절반(0.125%p)을 넘었으므로 PARTIAL, 0.05%p만 낮아졌다면 DIRECTION_ONLY입니다."
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

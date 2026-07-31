from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from app.config import settings
from app.schemas.resolve import ResolutionAssessment

_SYSTEM_PROMPT = (
    "당신은 정보 카드의 결과를 판정하는 Resolution Agent입니다. "
    "카드 발행 시점에 미리 정해진 평가 지표와 임계값을 기준으로, 작성자가 제출한 "
    "실제 결과·근거를 대조해서 판정하세요. 새로운 기준을 만들지 말고 주어진 "
    "평가 지표/임계값/성공·실패 조건과 실제 결과를 대조하는 역할만 하세요.\n\n"
    "먼저 실제 결과에서 평가 지표의 변화량(수치, 부호 포함)을 계산한 뒤, "
    "아래 순서대로 **하나씩 차례로** 확인하고 처음으로 조건을 만족하는 단계에서 멈추세요:\n"
    "1. 변화량의 절대값이 완전적중 임계값 이상이고 예측한 방향과 일치한다 → SUCCESS\n"
    "2. (1이 아니고) 변화량의 절대값이 부분적중 임계값 이상이고 예측한 방향과 일치한다 → PARTIAL\n"
    "   (부분적중 임계값 이상 완전적중 임계값 미만 구간은 반드시 PARTIAL이며 DIRECTION_ONLY가 아닙니다)\n"
    "3. (1, 2가 아니고) 변화량이 0보다 크고(예측한 방향과 부호는 같음) 부분적중 임계값 미만이다 → DIRECTION_ONLY\n"
    "4. 변화량의 방향이 예측과 반대다 → FAILURE\n"
    "5. 실제 결과·근거가 평가 지표와 무관하거나 판단할 근거가 부족하다 → INVALID\n\n"
    "예시: 완전적중 임계값 2, 부분적중 임계값 0.5인 상승 예측 카드에서 실제 변화량이 +1.0이면, "
    "0.5 이상 2 미만 구간이므로 PARTIAL입니다."
)


class ResolutionState(TypedDict):
    claim: str
    success_condition: str
    failure_condition: str
    actual_result: str
    evidence_summary: str
    evaluation_metric: str
    full_hit_threshold: float
    partial_hit_threshold: float
    verdict: str
    ai_reason: str


def _resolve(state: ResolutionState) -> dict:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not configured")

    human_content = (
        f"주장: {state['claim']}\n"
        f"성공 조건: {state['success_condition']}\n"
        f"실패 조건: {state['failure_condition']}\n"
        f"평가 지표: {state['evaluation_metric']}\n"
        f"완전적중 임계값: {state['full_hit_threshold']}\n"
        f"부분적중 임계값: {state['partial_hit_threshold']}\n\n"
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
    evaluation_metric: str,
    full_hit_threshold: float,
    partial_hit_threshold: float,
) -> dict:
    return _resolution_graph.invoke(
        {
            "claim": claim,
            "success_condition": success_condition,
            "failure_condition": failure_condition,
            "actual_result": actual_result,
            "evidence_summary": evidence_summary,
            "evaluation_metric": evaluation_metric,
            "full_hit_threshold": full_hit_threshold,
            "partial_hit_threshold": partial_hit_threshold,
            "verdict": "",
            "ai_reason": "",
        }
    )

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from app.agents.counter_agent import run_counter_agent
from app.agents.evidence_agent import run_evidence_agent
from app.config import settings
from app.schemas.verify import SynthesisAssessment

_SYNTHESIS_SYSTEM_PROMPT = (
    "당신은 정보 카드의 최종 심사를 종합하는 에이전트입니다. Evidence Agent와 "
    "Counter Agent의 평가를 참고해 카드 제목/요약을 생성하고, 원화 기준 추천 판매가 "
    "범위(5,000~50,000원 사이), 기존에 널리 알려진 정보와의 중복도(0~100), 최종 "
    "통과 여부(PASSED/FAILED)를 결정하세요. 출처 연결성이나 반대 논리 평가가 "
    "LOW 이하이면 FAILED로 판정하세요."
)


class VerifyState(TypedDict):
    claim: str
    success_condition: str
    failure_condition: str
    evidence_summary: str
    category: str | None
    sources: list[dict]

    evidence_relevance_level: str
    evidence_relevance_comment: str

    counterargument_level: str
    counterargument_comment: str
    missing_variable_level: str
    missing_variable_comment: str

    status: str
    generated_title: str
    generated_summary: str
    recommended_price_min: int
    recommended_price_max: int
    duplication_score: float
    overall_comment: str


def _evidence_node(state: VerifyState) -> dict:
    assessment = run_evidence_agent(
        state["claim"],
        state["success_condition"],
        state["failure_condition"],
        state["sources"],
    )
    return {
        "evidence_relevance_level": assessment.evidence_relevance_level,
        "evidence_relevance_comment": assessment.evidence_relevance_comment,
    }


def _counter_node(state: VerifyState) -> dict:
    assessment = run_counter_agent(
        state["claim"],
        state["success_condition"],
        state["failure_condition"],
        state["evidence_summary"],
    )
    return {
        "counterargument_level": assessment.counterargument_level,
        "counterargument_comment": assessment.counterargument_comment,
        "missing_variable_level": assessment.missing_variable_level,
        "missing_variable_comment": assessment.missing_variable_comment,
    }


def _synthesize_node(state: VerifyState) -> dict:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not configured")

    human_content = (
        f"주장: {state['claim']}\n"
        f"성공 조건: {state['success_condition']}\n"
        f"실패 조건: {state['failure_condition']}\n"
        f"근거 요약: {state['evidence_summary']}\n"
        f"카테고리: {state.get('category') or '미지정'}\n\n"
        "[Evidence Agent 평가]\n"
        f"- 출처 연결성: {state['evidence_relevance_level']} — "
        f"{state['evidence_relevance_comment']}\n\n"
        "[Counter Agent 평가]\n"
        f"- 반대 논리: {state['counterargument_level']} — "
        f"{state['counterargument_comment']}\n"
        f"- 누락 변수: {state['missing_variable_level']} — "
        f"{state['missing_variable_comment']}"
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.groq_api_key)
    structured_llm = llm.with_structured_output(SynthesisAssessment)
    result = structured_llm.invoke(
        [SystemMessage(content=_SYNTHESIS_SYSTEM_PROMPT), HumanMessage(content=human_content)]
    )
    return result.model_dump()


def build_verify_graph():
    graph = StateGraph(VerifyState)
    graph.add_node("evidence", _evidence_node)
    graph.add_node("counter", _counter_node)
    graph.add_node("synthesize", _synthesize_node)

    graph.add_edge(START, "evidence")
    graph.add_edge(START, "counter")
    graph.add_edge("evidence", "synthesize")
    graph.add_edge("counter", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


_verify_graph = build_verify_graph()


def run_verify_graph(
    claim: str,
    success_condition: str,
    failure_condition: str,
    evidence_summary: str,
    category: str | None,
    sources: list[dict],
) -> dict:
    return _verify_graph.invoke(
        {
            "claim": claim,
            "success_condition": success_condition,
            "failure_condition": failure_condition,
            "evidence_summary": evidence_summary,
            "category": category,
            "sources": sources,
        }
    )

from typing import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.config import settings
from app.schemas.draft_assist import DraftAssistAction

_SYSTEM_PROMPTS = {
    DraftAssistAction.CLARIFY: (
        "당신은 정보 카드 작성을 돕는 도우미입니다. "
        "사용자가 입력한 주장을 더 명확하고 간결한 한 문장으로 다듬어주세요. "
        "다듬은 문장만 출력하고 다른 설명은 하지 마세요."
    ),
    DraftAssistAction.VERIFIABLE_REWRITE: (
        "당신은 정보 카드 작성을 돕는 도우미입니다. "
        "사용자가 입력한 주장을 나중에 사실 여부를 객관적으로 판정할 수 있는 "
        "검증 가능한 문장(구체적인 수치, 기간, 기준 포함)으로 다시 써주세요. "
        "다시 쓴 문장만 출력하고 다른 설명은 하지 마세요."
    ),
    DraftAssistAction.SUGGEST_VARIABLES: (
        "당신은 정보 카드 작성을 돕는 도우미입니다. "
        "사용자가 입력한 주장을 보고, 나중에 결과 판정에 영향을 줄 수 있는데 "
        "빠져 있는 핵심 변수(기간, 수치 임계값, 판정 기준 등)를 목록으로 제안해주세요."
    ),
    DraftAssistAction.EXAMPLE: (
        "당신은 정보 카드 작성을 돕는 도우미입니다. "
        "주어진 카테고리에 어울리는, 좋은 정보 카드의 예시 주장 문장을 2~3개 보여주세요."
    ),
}


class RefinementState(TypedDict):
    action: DraftAssistAction
    text: str
    category: str | None
    result: str


def _refine(state: RefinementState) -> RefinementState:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not configured")

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.groq_api_key)
    system_prompt = _SYSTEM_PROMPTS[state["action"]]

    human_parts = []
    if state.get("category"):
        human_parts.append(f"카테고리: {state['category']}")
    if state.get("text"):
        human_parts.append(f"입력 문장: {state['text']}")
    human_content = "\n".join(human_parts) or "(입력 없음)"

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
    )
    return {**state, "result": response.content}


def build_refinement_graph():
    graph = StateGraph(RefinementState)
    graph.add_node("refine", _refine)
    graph.set_entry_point("refine")
    graph.add_edge("refine", END)
    return graph.compile()


_refinement_graph = build_refinement_graph()


def run_refinement_agent(action: DraftAssistAction, text: str, category: str | None) -> str:
    result = _refinement_graph.invoke(
        {"action": action, "text": text, "category": category, "result": ""}
    )
    return result["result"]

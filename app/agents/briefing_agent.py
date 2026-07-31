from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from app.config import settings
from app.schemas.briefing import BriefingAssessment

_SYSTEM_PROMPT = (
    "당신은 정보 카드 플랫폼의 일일 브리핑을 작성하는 에이전트입니다. "
    "최근 발행된 정보 카드들의 주장을 분석해서, 오늘 시장에서 주목할 만한 핵심 "
    "트렌드 3가지를 15자 내외의 짧은 문구로 요약하고, 각각의 방향성(상승/하락/중립)을 "
    "판단하세요. 여러 카드에 걸쳐 반복되거나 공통된 주제를 우선하세요."
)


class BriefingState(TypedDict):
    cards: list[dict]
    insights: list[dict]


def _generate(state: BriefingState) -> dict:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not configured")

    if state["cards"]:
        card_lines = "\n".join(
            f"- [{card.get('category') or '미지정'}] {card['claim']}" for card in state["cards"]
        )
    else:
        card_lines = "(최근 발행된 카드 없음)"

    human_content = f"최근 발행된 정보 카드 주장 목록:\n{card_lines}"

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.groq_api_key)
    structured_llm = llm.with_structured_output(BriefingAssessment)
    result = structured_llm.invoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_content)]
    )
    return {"insights": [insight.model_dump() for insight in result.insights]}


def build_briefing_graph():
    graph = StateGraph(BriefingState)
    graph.add_node("generate", _generate)
    graph.set_entry_point("generate")
    graph.add_edge("generate", END)
    return graph.compile()


_briefing_graph = build_briefing_graph()


def run_briefing_agent(cards: list[dict]) -> dict:
    return _briefing_graph.invoke({"cards": cards, "insights": []})

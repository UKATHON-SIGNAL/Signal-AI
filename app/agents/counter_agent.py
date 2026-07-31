from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import settings
from app.schemas.verify import CounterAssessment

_SYSTEM_PROMPT = (
    "당신은 정보 카드의 반대 논리와 누락된 변수를 탐색하는 Counter Agent입니다. "
    "이 주장이 틀릴 수 있는 반대 근거를 찾고, 결과 판정에 영향을 줄 수 있는데 "
    "빠져 있는 핵심 변수(기간, 수치 임계값, 판정 기준 등)가 있는지 평가하세요."
)


def run_counter_agent(
    claim: str,
    success_condition: str,
    failure_condition: str,
    evidence_summary: str,
) -> CounterAssessment:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not configured")

    human_content = (
        f"주장: {claim}\n"
        f"성공 조건: {success_condition}\n"
        f"실패 조건: {failure_condition}\n"
        f"근거 요약: {evidence_summary}"
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.groq_api_key)
    structured_llm = llm.with_structured_output(CounterAssessment)
    return structured_llm.invoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_content)]
    )

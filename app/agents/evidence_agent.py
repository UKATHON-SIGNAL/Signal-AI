import requests
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import settings
from app.schemas.verify import EvidenceAssessment

_SYSTEM_PROMPT = (
    "당신은 정보 카드의 출처를 검증하는 Evidence Agent입니다. "
    "주어진 주장과, 실제로 가져온 출처 본문 내용을 비교해서 출처가 주장을 "
    "얼마나 뒷받침하는지 평가하세요. 출처에 접근하지 못했다면 그 사실을 반영해 "
    "신뢰도를 낮게 평가하세요."
)


def _fetch_source_text(url: str, max_chars: int = 2000) -> str:
    try:
        response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:max_chars] or "(본문 텍스트를 추출하지 못함)"
    except Exception as e:
        return f"[출처 접근 실패: {e}]"


def run_evidence_agent(
    claim: str,
    success_condition: str,
    failure_condition: str,
    sources: list[dict],
) -> EvidenceAssessment:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not configured")

    if not sources:
        source_block = "(제공된 출처 없음)"
    else:
        fetched = []
        for source in sources:
            content = _fetch_source_text(source["url"])
            label = source.get("title") or source["url"]
            fetched.append(f"- {label} ({source['url']}):\n{content}")
        source_block = "\n\n".join(fetched)

    human_content = (
        f"주장: {claim}\n"
        f"성공 조건: {success_condition}\n"
        f"실패 조건: {failure_condition}\n\n"
        f"출처 내용:\n{source_block}"
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.groq_api_key)
    structured_llm = llm.with_structured_output(EvidenceAssessment)
    return structured_llm.invoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_content)]
    )

# Signal Market — Signal-AI

**Signal Market**의 AI 에이전트 서버입니다. [Signal-BE](../Signal-BE)(Spring Boot)가 카드 검증/가격 산정/결과 판정/브리핑 생성이 필요할 때마다 HTTP로 호출하는 내부 API 서버로, LangGraph 기반 에이전트 그래프와 Groq(Llama 3.3 70B, 무료 오픈소스 모델)를 사용합니다.

```
Signal-BE (Spring Boot) → Signal-AI (FastAPI + LangGraph + Groq)
```

## 에이전트 구성

| 에이전트 | 파일 | 역할 |
| --- | --- | --- |
| Refinement | `app/agents/refinement_agent.py` | 정보 카드 초안 작성을 돕는 어시스트(명확화/검증가능한 문장으로 재작성/변수 제안/예시) |
| Evidence | `app/agents/evidence_agent.py` | 출처 URL을 실제로 크롤링(requests+BeautifulSoup)해서 근거와 주장의 연결성 평가 |
| Counter | `app/agents/counter_agent.py` | 반대 논리·누락 변수 평가 |
| **Verify Graph** | `app/agents/verify_graph.py` | Evidence·Counter를 병렬 실행 후 Synthesis 노드가 종합 — 작성자 성과/출처 신뢰도/정보 유효기간/실제 카드 중복도(Signal-BE가 계산해 전달)까지 반영해 PASSED/FAILED와 추천가 산출 |
| Resolution | `app/agents/resolution_agent.py` | 결과 확정 시 실제 결과와 판정 조건을 대조해 완전적중/부분적중/방향만적중/실패/판정불가 5단계로 판정 |
| Briefing | `app/agents/briefing_agent.py` | 최근 발행 카드들을 분석해 홈 화면용 인사이트 3건(제목+추세) 생성 |

`verify_graph`는 `evidence`/`counter` 노드를 병렬로 실행한 뒤 `synthesize` 노드에서 결과를 합치는 LangGraph `StateGraph`로 구성되어 있습니다.

## API

| Method | URL | 설명 |
| --- | --- | --- |
| POST | `/api/draft-assist` | 정보 카드 작성 보조(명확화/재작성/변수제안/예시) |
| POST | `/api/verify` | 카드 AI 검토 (Evidence + Counter + Synthesis) |
| POST | `/api/resolve` | 결과 판정 (5단계 Resolution) |
| POST | `/api/briefing` | 오늘의 AI 브리핑 생성 |
| GET | `/health` | 헬스체크 |

각 요청/응답 스키마는 `app/schemas/`, 라우터는 `app/api/`에 정의되어 있습니다. Signal-BE 쪽에서 이 API들을 호출하는 클라이언트/DTO는 `Signal-BE/src/main/java/com/signal/signalbe/client/signalai`에 있습니다.

## 로컬 실행

### 요구사항
- Python 3.12
- Groq API Key ([console.groq.com](https://console.groq.com)에서 무료 발급)

### 1. 가상환경 및 의존성 설치
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일 생성:
```
GROQ_API_KEY=your-groq-api-key
```

### 3. 서버 실행
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```
기본적으로 `http://localhost:8001`에서 뜨며, Signal-BE는 `SIGNAL_AI_BASE_URL` 환경 변수로 이 주소를 바라봅니다.

### 환경 변수

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `GROQ_API_KEY` | (없음) | Groq API 키. 미설정 시 `/api/verify`, `/api/resolve`, `/api/briefing` 호출 시 503 반환 |
| `PORT` | `8000` | 서버 포트 (배포 환경에서 플랫폼이 주입) |

## 배포

Railway에 별도 서비스로 배포되어 있으며 `main` 브랜치 push 시 자동 재배포됩니다.

- `https://signal-ai-production-d7dd.up.railway.app`

## 기술 스택

- Python 3.12, FastAPI, Uvicorn
- LangGraph, LangChain, langchain-groq (Llama 3.3 70B)
- Pydantic / pydantic-settings
- requests, BeautifulSoup4 (출처 크롤링)
- Docker / Railway

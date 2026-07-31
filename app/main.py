from fastapi import FastAPI

from app.api.draft_assist import router as draft_assist_router
from app.api.verify import router as verify_router

app = FastAPI(title="Signal-AI")
app.include_router(draft_assist_router)
app.include_router(verify_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

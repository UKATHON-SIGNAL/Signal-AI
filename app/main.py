from fastapi import FastAPI

from app.api.draft_assist import router as draft_assist_router

app = FastAPI(title="Signal-AI")
app.include_router(draft_assist_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

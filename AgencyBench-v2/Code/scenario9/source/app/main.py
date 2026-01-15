from fastapi import FastAPI

from app.webhooks.github import router as github_webhook_router
from app.lifecycle import lifespan


app = FastAPI(title="Agent MVP Service", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


app.include_router(github_webhook_router)



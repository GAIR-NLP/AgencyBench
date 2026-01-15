from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db.mongo import init_indexes


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    init_indexes()
    try:
        yield
    finally:
        # shutdown
        pass



import asyncio
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as redis
from fastapi import FastAPI
from fastembed import SparseTextEmbedding
from google import genai
from qdrant_client import AsyncQdrantClient

from app.api.routes import router
from app.core.settings import get_settings
from app.core.state import AppState

from app.core.observability import langfuse

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    print("[startup] Initializing Gemini async client...")
    gemini = genai.Client(api_key=settings.gemini_api_key)

    print("[startup] Initializing Async Qdrant client...")
    qdrant = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    print("[startup] Loading local BM25 model...")
    bm25 = SparseTextEmbedding(model_name=settings.sparse_model)

    print("[startup] Initializing shared HTTP client...")
    http = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.jina_timeout_seconds),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )

    print("[startup] Connecting to Redis...")
    redis_client = redis.from_url(
        settings.redis_url,
        decode_responses=True,
    )
    await redis_client.ping()

    print("[startup] Checking Qdrant collection...")
    await qdrant.get_collection(settings.collection_name)

    app.state.rag_state = AppState(
        gemini=gemini,
        qdrant=qdrant,
        bm25=bm25,
        http=http,
        redis=redis_client,
        subquery_semaphore=asyncio.Semaphore(settings.max_subquery_concurrency),
        sparse_semaphore=asyncio.Semaphore(settings.sparse_thread_concurrency),
    )

    print("[startup] FastAPI RAG application is ready.")

    try:
        yield
    finally:
        print("[shutdown] Closing clients...")
        await http.aclose()
        await qdrant.close()
        await redis_client.aclose()
        await gemini.aio.aclose()

        if langfuse is not None:
            langfuse.shutdown()

        print("[shutdown] Done.")


app = FastAPI(
    title="Multi-Hop RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "multihop-rag",
        "docs": "/docs",
        "health": "/api/v1/health",
        "query": "/api/v1/query",
    }

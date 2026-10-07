import time

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.schemas import QueryRequest, RAGResponse
from app.core.settings import Settings, get_settings
from app.core.state import AppState
from app.pipeline.rag_service import ask_rag_with_decomposition

router = APIRouter(prefix="/api/v1", tags=["RAG"])


def get_app_state(request: Request) -> AppState:
    return request.app.state.rag_state


def get_app_settings() -> Settings:
    return get_settings()


@router.get("/health")
async def health(state: AppState = Depends(get_app_state)):
    redis_ok = False
    qdrant_ok = False

    try:
        redis_ok = bool(await state.redis.ping())
    except Exception:
        pass

    try:
        await state.qdrant.get_collection(get_settings().collection_name)
        qdrant_ok = True
    except Exception:
        pass

    return {
        "status": "ok" if redis_ok and qdrant_ok else "degraded",
        "redis": redis_ok,
        "qdrant": qdrant_ok,
    }


@router.post("/query", response_model=RAGResponse)
async def query_rag(
    payload: QueryRequest,
    state: AppState = Depends(get_app_state),
    settings: Settings = Depends(get_app_settings),
):
    started = time.perf_counter()
    try:
        result = await ask_rag_with_decomposition(
            payload.query,
            state,
            settings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if result.get("cache_hit"):
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)

    return result

import asyncio

from qdrant_client import models

from app.core.observability import observe
from app.core.settings import Settings
from app.core.state import AppState


def _create_sparse_embedding_sync(text: str, state: AppState) -> models.SparseVector:
    result = list(state.bm25.embed([text]))[0]
    return models.SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


@observe(
    name="sparse-query-embedding",
    as_type="embedding",
    capture_input=True,
    capture_output=False,
)
async def create_sparse_embedding(text: str, state: AppState, settings: Settings) -> models.SparseVector:
    # FastEmbed is local CPU work. Offload it so it does not block the FastAPI event loop.
    async with state.sparse_semaphore:
        return await asyncio.to_thread(_create_sparse_embedding_sync, text, state)

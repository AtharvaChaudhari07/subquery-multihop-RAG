import numpy as np

from qdrant_client import models

from app.core.observability import observe
from app.core.settings import Settings
from app.core.state import AppState


@observe(
    name="hybrid-qdrant-retrieval",
    as_type="retriever",
    capture_input=False,
    capture_output=True,
)
async def hybrid_retrieve(
    dense_query: np.ndarray,
    sparse_query: models.SparseVector,
    source_filter,
    state: AppState,
    settings: Settings,
):
    # Your current Qdrant design is dense + BM25 prefetch followed by RRF.
    # Therefore dense/sparse embedding creation runs in parallel BEFORE this call.
    result = await state.qdrant.query_points(
        collection_name=settings.collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_query.tolist(),
                using="dense",
                limit=settings.dense_retrieval_limit,
                filter=source_filter,
            ),
            models.Prefetch(
                query=sparse_query,
                using="bm25",
                limit=settings.sparse_retrieval_limit,
                filter=source_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=settings.hybrid_limit,
        with_payload=True,
    )
    return result.points

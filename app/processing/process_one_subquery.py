import asyncio

from app.core.observability import observe
from app.core.settings import Settings
from app.core.state import AppState
from app.core.schemas import SubQuery
from app.embeddings.dense_embedding import create_dense_embedding
from app.embeddings.sparse_embedding import create_sparse_embedding
from app.retrieval.extract_candidates import extract_candidates
from app.retrieval.hybrid_retrieve import hybrid_retrieve
from app.retrieval.source_filter import build_source_filter
from app.reranking.rerank_with_jina import rerank_with_jina


@observe(
    name="subquery-processing",
    as_type="chain",
    capture_input=True,
    capture_output=False,
)
async def process_one_subquery(
    subquery_obj: SubQuery,
    state: AppState,
    settings: Settings,
) -> dict:
    async with state.subquery_semaphore:
        # IMPORTANT:
        # 1. all subqueries run concurrently in process_all_subqueries()
        # 2. inside each subquery, dense and sparse embedding generation runs concurrently
        dense_query, sparse_query = await asyncio.gather(
            create_dense_embedding(subquery_obj.query, state, settings),
            create_sparse_embedding(subquery_obj.query, state, settings),
        )

        source_filter = build_source_filter(subquery_obj.source)

        points = await hybrid_retrieve(
            dense_query=dense_query,
            sparse_query=sparse_query,
            source_filter=source_filter,
            state=state,
            settings=settings,
        )

        candidates = extract_candidates(
            points=points,
            subquery_id=subquery_obj.id,
            subquery=subquery_obj.query,
            source=subquery_obj.source,
        )

        reranked = await rerank_with_jina(
            query=subquery_obj.query,
            candidates=candidates,
            state=state,
            settings=settings,
        )

        return {
            "subquery_id": subquery_obj.id,
            "constraint": subquery_obj.constraint,
            "query": subquery_obj.query,
            "source": subquery_obj.source,
            "candidates": candidates,
            "reranked": reranked,
        }

import time

from app.core.cache import (
    acquire_cache_lock,
    build_cache_key,
    get_cached_result,
    release_cache_lock,
    set_cached_result,
    wait_for_cache_fill,
)
from app.core.observability import observe
from app.core.settings import Settings
from app.core.state import AppState
from app.decomposition.decompose_query import decompose_query
from app.generation.generate_final_answer import generate_final_answer
from app.pipeline.build_final_context import build_final_context
from app.pipeline.build_final_evidence import build_final_evidence
from app.pipeline.deduplicate_evidence import deduplicate_evidence
from app.processing.process_all_subqueries import process_all_subqueries


@observe(
    name="rag-pipeline-with-redis-cache",
    as_type="chain",
    capture_input=True,
    capture_output=True,
)
async def ask_rag_with_decomposition(
    query: str,
    state: AppState,
    settings: Settings,
) -> dict:
    query = " ".join(query.strip().split())
    if not query:
        raise ValueError("Query cannot be empty.")

    cache_key = build_cache_key(query, settings)

    # ------------------------------------------------------------
    # STEP 0 — FINAL ANSWER CACHE, BEFORE DECOMPOSITION
    # ------------------------------------------------------------
    cached = await get_cached_result(state.redis, cache_key)
    if cached is not None:
        return {
            "query": query,
            "answer": cached["answer"],
            "cache_hit": True,
            "subquery_count": cached.get("subquery_count", 0),
            "evidence_count": cached.get("evidence_count", 0),
        }

    # ------------------------------------------------------------
    # PREVENT CACHE STAMPEDE
    # ------------------------------------------------------------
    lock = await acquire_cache_lock(
        state.redis,
        cache_key,
        settings.lock_ttl_seconds,
    )

    if lock is None:
        # Another request is already computing exactly this query.
        cached = await wait_for_cache_fill(
            state.redis,
            cache_key,
            settings.lock_wait_timeout_seconds,
            settings.lock_poll_interval_seconds,
        )
        if cached is not None:
            return {
                "query": query,
                "answer": cached["answer"],
                "cache_hit": True,
                "subquery_count": cached.get("subquery_count", 0),
                "evidence_count": cached.get("evidence_count", 0),
            }

        # Re-check after timeout: the original worker may have released the lock.
        lock = await acquire_cache_lock(
            state.redis,
            cache_key,
            settings.lock_ttl_seconds,
        )
        if lock is None:
            raise RuntimeError("Timed out waiting for another request to populate the Redis cache.")

    lock_key, lock_token = lock

    try:
        # The cache may have been filled between the first GET and lock acquisition.
        cached = await get_cached_result(state.redis, cache_key)
        if cached is not None:
            return {
                "query": query,
                "answer": cached["answer"],
                "cache_hit": True,
                "subquery_count": cached.get("subquery_count", 0),
                "evidence_count": cached.get("evidence_count", 0),
            }

        started = time.perf_counter()

        # --------------------------------------------------------
        # STEP 1 — DECOMPOSE RAW USER QUERY
        # --------------------------------------------------------
        decomposition = await decompose_query(query, state, settings)

        # --------------------------------------------------------
        # STEP 2 — PARALLEL SUBQUERY FAN-OUT
        # --------------------------------------------------------
        all_subquery_results = await process_all_subqueries(
            decomposition,
            state,
            settings,
        )

        # --------------------------------------------------------
        # STEP 3 — EVIDENCE MERGE + DEDUP
        # --------------------------------------------------------
        final_evidence = build_final_evidence(
            all_subquery_results,
            settings.per_subquery_final_limit,
        )
        final_evidence = deduplicate_evidence(final_evidence)

        # --------------------------------------------------------
        # STEP 4 — FINAL CONTEXT
        # --------------------------------------------------------
        final_context = build_final_context(
            decomposition,
            final_evidence,
        )

        # --------------------------------------------------------
        # STEP 5 — FINAL GEMINI ANSWER
        # --------------------------------------------------------
        answer = await generate_final_answer(
            query,
            final_context,
            state,
            settings,
        )

        elapsed_ms = (time.perf_counter() - started) * 1000

        result = {
            "query": query,
            "answer": answer,
            "cache_hit": False,
            "subquery_count": len(decomposition.subqueries),
            "evidence_count": len(final_evidence),
            "latency_ms": round(elapsed_ms, 2),
        }

        # Store only the useful canonical result. Future cache-hit requests
        # add cache_hit=True at response time.
        await set_cached_result(
            state.redis,
            cache_key,
            {
                "answer": answer,
                "subquery_count": result["subquery_count"],
                "evidence_count": result["evidence_count"],
            },
            settings.cache_ttl_seconds,
        )

        return result

    finally:
        await release_cache_lock(state.redis, lock_key, lock_token)

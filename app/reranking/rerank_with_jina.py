import httpx

from app.core.observability import observe
from app.core.settings import Settings
from app.core.state import AppState


@observe(
    name="jina-reranking",
    as_type="span",
    capture_input=False,
    capture_output=True,
)
async def rerank_with_jina(
    query: str,
    candidates: list[dict],
    state: AppState,
    settings: Settings,
) -> list[dict]:
    if not candidates:
        return []

    documents = [candidate.get("chunk_embedding_text", "") for candidate in candidates]

    payload = {
        "model": settings.jina_reranker_model,
        "query": query,
        "top_n": min(settings.jina_top_n, len(documents)),
        "documents": documents,
        "return_documents": False,
        "max_doc_length": settings.jina_max_doc_length,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.jina_api_key}",
    }

    try:
        response = await state.http.post(
            settings.jina_rerank_url,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Jina reranker HTTP error {exc.response.status_code}: {exc.response.text[:1000]}"
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Jina reranker request failed: {exc}") from exc

    data = response.json()
    reranked: list[dict] = []

    for result in data.get("results", []):
        original_index = int(result["index"])
        candidate = candidates[original_index].copy()
        candidate["reranker_score"] = float(result["relevance_score"])
        reranked.append(candidate)

    return reranked

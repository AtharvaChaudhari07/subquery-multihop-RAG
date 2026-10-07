from app.core.observability import observe


@observe(
    name="extract-candidates",
    as_type="span",
    capture_input=False,
    capture_output=True,
)
def extract_candidates(points, subquery_id: str, subquery: str, source: str | None):
    candidates = []
    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}
        candidates.append(
            {
                "subquery_id": subquery_id,
                "subquery": subquery,
                "source_filter": source,
                "qdrant_rank": rank,
                "qdrant_score": float(point.score) if point.score is not None else None,
                "point_id": point.id,
                "chunk_id": payload.get("chunk_id"),
                "document_id": payload.get("document_id"),
                "chunk_embedding_text": payload.get("chunk_embedding_text", ""),
                "title": payload.get("title"),
                "author": payload.get("author"),
                "source": payload.get("source"),
                "published_at": payload.get("published_at"),
                "url": payload.get("url"),
                "citation": payload.get("citation"),
            }
        )
    return candidates

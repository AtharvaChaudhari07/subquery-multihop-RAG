from app.core.observability import observe


@observe(name="build-final-evidence", as_type="span")
def build_final_evidence(all_subquery_results: list[dict], per_subquery_limit: int) -> list[dict]:
    evidence: list[dict] = []

    for result in all_subquery_results:
        for item in result["reranked"][:per_subquery_limit]:
            evidence.append(
                {
                    "subquery_id": result["subquery_id"],
                    "constraint": result["constraint"],
                    "query": result["query"],
                    "source_filter": result["source"],
                    "reranker_score": item.get("reranker_score"),
                    "qdrant_rank": item.get("qdrant_rank"),
                    "qdrant_score": item.get("qdrant_score"),
                    "point_id": item.get("point_id"),
                    "chunk_id": item.get("chunk_id"),
                    "document_id": item.get("document_id"),
                    "title": item.get("title"),
                    "author": item.get("author"),
                    "source": item.get("source"),
                    "published_at": item.get("published_at"),
                    "url": item.get("url"),
                    "citation": item.get("citation"),
                    "text": item.get("chunk_embedding_text", ""),
                }
            )

    return evidence

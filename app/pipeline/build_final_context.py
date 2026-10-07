from app.core.observability import observe
from app.core.schemas import QueryDecomposition


@observe(
    name="build-final-context",
    as_type="span",
    capture_input=False,
    capture_output=True,
)
def build_final_context(
    decomposition: QueryDecomposition,
    final_evidence: list[dict],
) -> str:
    subquery_text = "\n".join(
        f"- {sq.id}: {sq.query} | source={sq.source} | constraint={sq.constraint}"
        for sq in decomposition.subqueries
    )

    evidence_parts = []
    for index, item in enumerate(final_evidence, start=1):
        matched = ", ".join(item.get("matched_subqueries", []))
        evidence_parts.append(
            f"""EVIDENCE {index}
Source: {item.get('source')}
Title: {item.get('title')}
Author: {item.get('author')}
Published: {item.get('published_at')}
Chunk ID: {item.get('chunk_id')}
Document ID: {item.get('document_id')}
Matched Subqueries: {matched}
Qdrant Rank: {item.get('qdrant_rank')}
Qdrant Score: {item.get('qdrant_score')}
Reranker Score: {item.get('reranker_score')}
URL: {item.get('url')}

TEXT:
{item.get('text', '')}"""
        )

    evidence_text = "\n\n".join(evidence_parts)
    return (
        "GENERATED SUBQUERIES:\n"
        f"{subquery_text}\n\n"
        "RETRIEVED EVIDENCE:\n"
        f"{evidence_text}"
    )

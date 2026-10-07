from app.core.observability import observe


@observe(name="deduplicate-evidence", as_type="span")
def deduplicate_evidence(evidence: list[dict]) -> list[dict]:
    unique: dict[object, dict] = {}

    for item in evidence:
        key = (
            item.get("chunk_id")
            or item.get("document_id")
            or (
                item.get("source"),
                item.get("title"),
                item.get("text", "")[:100],
            )
        )

        if key not in unique:
            copy = item.copy()
            copy["matched_subqueries"] = [item["subquery_id"]]
            unique[key] = copy
        else:
            existing = unique[key]
            if item["subquery_id"] not in existing["matched_subqueries"]:
                existing["matched_subqueries"].append(item["subquery_id"])

    return list(unique.values())

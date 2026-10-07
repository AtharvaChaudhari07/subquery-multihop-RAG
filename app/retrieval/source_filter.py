from qdrant_client import models

from app.core.observability import observe


@observe(name="build-source-filter", as_type="span")
def build_source_filter(source: str | None):
    if source is None:
        return None

    return models.Filter(
        must=[
            models.FieldCondition(
                key="source",
                match=models.MatchValue(value=source),
            )
        ]
    )

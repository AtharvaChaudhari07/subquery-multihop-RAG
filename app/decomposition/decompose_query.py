from google.genai import types

from app.core.observability import observe
from app.core.schemas import QueryDecomposition
from app.core.settings import Settings
from app.core.sources import AVAILABLE_SOURCES, AVAILABLE_SOURCES_TEXT
from app.core.state import AppState


@observe(
    name="query-decomposition",
    as_type="generation",
    capture_input=True,
    capture_output=True,
)
async def decompose_query(query: str, state: AppState, settings: Settings) -> QueryDecomposition:
    prompt = f"""
You are a query decomposition system for a retrieval-augmented question answering benchmark.

The benchmark contains MULTI-CONSTRAINT / INTERSECTION questions.
The questions generally ask for one entity that satisfies multiple independent conditions
that may be documented across different documents.

Your job is ONLY to decompose the question into independent retrieval subqueries.
Do NOT answer the original question.
Do NOT perform sequential/iterative reasoning.
Each subquery must be independently searchable.

============================================================
AVAILABLE SOURCES
============================================================

The document corpus contains ONLY the following source names:

{AVAILABLE_SOURCES_TEXT}

============================================================
SOURCE EXTRACTION RULES
============================================================

For every subquery:
1. Determine whether the constraint explicitly refers to a source/publication.
2. If it refers to one of the available sources, normalize the source name to the EXACT corresponding value.
3. If the question mentions a source that is NOT present in the available source list, set source = null.
4. If the subquery does not have a source constraint, set source = null.
5. NEVER invent a source.
6. NEVER return a source value that is not exactly present in the available source list.
7. The source field is metadata for retrieval filtering. The query itself should still be useful standalone retrieval text.

============================================================
DECOMPOSITION RULES
============================================================

Create independent subqueries for independent constraints.

Do not make one subquery depend on the answer from another.
Do not create "find the person" followed by "search that person".
All queries must be independently searchable.
Preserve named entities, organizations, events, and distinctive facts from the original question.
Do not invent facts.
Do not create unnecessary redundant queries.

============================================================
ORIGINAL QUESTION
============================================================

{query}
"""

    response = await state.gemini.aio.models.generate_content(
        model=settings.decomposition_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=QueryDecomposition,
            temperature=settings.decomposition_temperature,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty decomposition response.")

    decomposition = QueryDecomposition.model_validate_json(response.text)

    valid_sources = set(AVAILABLE_SOURCES)
    for subquery in decomposition.subqueries:
        if subquery.source is not None and subquery.source not in valid_sources:
            subquery.source = None

    return decomposition

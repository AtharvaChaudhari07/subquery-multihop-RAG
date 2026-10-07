from google.genai import types

from app.core.observability import observe
from app.core.settings import Settings
from app.core.state import AppState


@observe(
    name="final-answer-generation",
    as_type="generation",
    capture_input=False,
    capture_output=True,
)
async def generate_final_answer(
    query: str,
    final_context: str,
    state: AppState,
    settings: Settings,
) -> str:
    final_prompt = f"""
You are a retrieval-grounded question answering system.

Answer the user's question using ONLY the retrieved evidence provided below.

The question is a MULTI-CONSTRAINT / INTERSECTION question.
The answer should be an entity that satisfies the relevant independent constraints together.

============================================================
IMPORTANT SOURCE RULE
============================================================

The corpus has a fixed set of available sources.
If the user explicitly requested a source that is NOT present in the available source list,
retrieval could not apply a source filter for that source.
Do NOT pretend that evidence from another source is evidence from the unavailable source.
If the requested source is unavailable and this prevents the question from being confidently answered,
explicitly say so.

============================================================
INTERSECTION REASONING
============================================================

1. Examine every independent constraint.
2. Determine what evidence supports each constraint.
3. Determine whether the SAME entity satisfies the constraints.
4. Do not merge evidence about different people/entities.
5. Do not select an entity merely because it appears frequently.
6. Do not use outside knowledge.
7. Do not treat generated subqueries as evidence.
8. The retrieved documents are the only evidence you may use.
9. If evidence is insufficient, say that it is insufficient.
10. Cite important claims using source, title, and chunk ID.

============================================================
USER QUESTION
============================================================

{query}

============================================================
RETRIEVED INFORMATION
============================================================

{final_context}

============================================================
ANSWER
============================================================

Give the direct answer first.
Then briefly explain how the evidence satisfies the independent constraints.
If one or more constraints cannot be established from the retrieved evidence,
explicitly identify that limitation.
"""

    response = await state.gemini.aio.models.generate_content(
        model=settings.generation_model,
        contents=final_prompt,
        config=types.GenerateContentConfig(
            temperature=settings.final_answer_temperature,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty final answer.")

    return response.text.strip()

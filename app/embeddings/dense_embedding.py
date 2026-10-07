import numpy as np
from google.genai import types

from app.core.observability import observe
from app.core.settings import Settings
from app.core.state import AppState


@observe(
    name="dense-query-embedding",
    as_type="embedding",
    capture_input=True,
    capture_output=False,
)
async def create_dense_embedding(text: str, state: AppState, settings: Settings) -> np.ndarray:
    result = await state.gemini.aio.models.embed_content(
        model=settings.dense_embedding_model,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=settings.dense_embedding_dim,
        ),
    )

    if not result.embeddings:
        raise RuntimeError("Gemini returned no dense embeddings.")

    vector = np.asarray(result.embeddings[0].values, dtype=np.float32)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    return vector

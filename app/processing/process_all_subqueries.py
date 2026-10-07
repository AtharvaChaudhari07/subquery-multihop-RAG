import asyncio

from app.core.observability import observe
from app.core.schemas import QueryDecomposition
from app.core.settings import Settings
from app.core.state import AppState
from app.processing.process_one_subquery import process_one_subquery


@observe(
    name="process-all-subqueries",
    as_type="chain",
    capture_input=True,
    capture_output=False,
)
async def process_all_subqueries(
    decomposition: QueryDecomposition,
    state: AppState,
    settings: Settings,
) -> list[dict]:
    # Independent subqueries are fan-out work, so run them together.
    tasks = [
        process_one_subquery(subquery, state, settings)
        for subquery in decomposition.subqueries
    ]
    return await asyncio.gather(*tasks)

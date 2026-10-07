from typing import Optional

from pydantic import BaseModel, Field


class SubQuery(BaseModel):
    id: str = Field(description="Unique subquery ID such as q1, q2, q3")
    query: str = Field(
        description="Standalone retrieval query containing the information needed to retrieve evidence for this constraint."
    )
    source: Optional[str] = Field(
        default=None,
        description="Normalized source name, or null when no source filter applies.",
    )
    constraint: str = Field(description="Independent condition represented by this subquery")


class QueryDecomposition(BaseModel):
    subqueries: list[SubQuery] = Field(description="List of all independent subqueries")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


class RAGResponse(BaseModel):
    query: str
    answer: str
    cache_hit: bool
    subquery_count: int
    evidence_count: int
    latency_ms: float

import asyncio
from dataclasses import dataclass

import httpx
import redis.asyncio as redis
from fastembed import SparseTextEmbedding
from google import genai
from qdrant_client import AsyncQdrantClient


@dataclass
class AppState:
    gemini: genai.Client
    qdrant: AsyncQdrantClient
    bm25: SparseTextEmbedding
    http: httpx.AsyncClient
    redis: redis.Redis
    subquery_semaphore: asyncio.Semaphore
    sparse_semaphore: asyncio.Semaphore

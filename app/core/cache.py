import asyncio
import hashlib
import json
import time
import uuid

from app.core.observability import observe
from app.core.settings import Settings
from redis.asyncio import Redis

_RELEASE_LOCK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
"""


@observe(name="redis-cache-key", as_type="span")
def build_cache_key(query: str, settings: Settings) -> str:
    normalized = " ".join(query.strip().split()).lower()
    query_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"rag:{settings.rag_cache_version}:answer:{query_hash}"


@observe(name="redis-cache-get", as_type="retriever")
async def get_cached_result(redis: Redis, key: str) -> dict | None:
    value = await redis.get(key)
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


@observe(name="redis-cache-set", as_type="span")
async def set_cached_result(
    redis: Redis,
    key: str,
    value: dict,
    ttl_seconds: int,
) -> None:
    await redis.set(
        key,
        json.dumps(value, ensure_ascii=False),
        ex=ttl_seconds,
    )


async def acquire_cache_lock(redis: Redis, cache_key: str, ttl_seconds: int) -> tuple[str, str] | None:
    lock_key = f"{cache_key}:lock"
    token = uuid.uuid4().hex
    acquired = await redis.set(
        lock_key,
        token,
        ex=ttl_seconds,
        nx=True,
    )
    if acquired:
        return lock_key, token
    return None


async def release_cache_lock(redis: Redis, lock_key: str, token: str) -> None:
    await redis.eval(_RELEASE_LOCK_SCRIPT, 1, lock_key, token)


async def wait_for_cache_fill(
    redis: Redis,
    cache_key: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = await get_cached_result(redis, cache_key)
        if result is not None:
            return result
        await asyncio.sleep(poll_interval_seconds)
    return None

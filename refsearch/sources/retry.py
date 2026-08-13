import asyncio
import random


async def with_backoff(coro_fn, *, retries: int = 4, base_delay: float = 1.0, retry_statuses=(429, 500, 502, 503, 504)):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            response = await coro_fn()
        except Exception as exc:
            last_exc = exc
            response = None
        else:
            if response.status_code not in retry_statuses:
                return response
            last_exc = RuntimeError(f"HTTP {response.status_code}")
        if attempt < retries:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            await asyncio.sleep(delay)
    if last_exc:
        raise last_exc
    return response

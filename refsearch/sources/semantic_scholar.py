import asyncio
import time

import httpx

from refsearch.models import Paper
from refsearch.sources.retry import with_backoff

_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_REFERENCES_URL_TMPL = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
_CITATIONS_URL_TMPL = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
_FIELDS = "title,authors,year,abstract,venue,externalIds,url,fieldsOfStudy"
_REFERENCES_FIELDS = "contexts,intents,title,externalIds,year,authors,venue,abstract"
_CITATIONS_FIELDS = "contexts,intents,title,externalIds,year,authors,venue,abstract"

# S2's per-key rate limit is roughly 1 request/second; without client-side
# spacing, concurrent pipeline calls self-inflict 429s during batch/eval runs.
_MIN_DELAY = 1.0
_last_call_lock = asyncio.Lock()
_last_call_time = 0.0


async def _throttle():
    global _last_call_time
    async with _last_call_lock:
        elapsed = time.monotonic() - _last_call_time
        if elapsed < _MIN_DELAY:
            await asyncio.sleep(_MIN_DELAY - elapsed)
        _last_call_time = time.monotonic()


async def search(
    client: httpx.AsyncClient,
    query: str,
    *,
    api_key: str | None = None,
    field: str | None = None,
    limit: int = 20,
) -> list[Paper]:
    params = {"query": query, "limit": limit, "fields": _FIELDS}
    if field:
        params["fieldsOfStudy"] = field
    headers = {"x-api-key": api_key} if api_key else {}

    await _throttle()

    async def call():
        return await client.get(_API_URL, params=params, headers=headers, timeout=20)

    response = await with_backoff(call)
    if response.status_code != 200:
        return []
    data = response.json()
    papers = []
    for item in data.get("data", []):
        external_ids = item.get("externalIds") or {}
        papers.append(
            Paper(
                title=item.get("title") or "",
                authors=[a.get("name", "") for a in item.get("authors") or []],
                year=item.get("year"),
                venue=item.get("venue") or "",
                doi=external_ids.get("DOI"),
                abstract=item.get("abstract") or "",
                url=item.get("url") or "",
                source="semantic_scholar",
                fields_of_study=item.get("fieldsOfStudy") or [],
            )
        )
    return papers


async def get_references(
    client: httpx.AsyncClient,
    paper_id: str,
    *,
    api_key: str | None = None,
    limit: int = 1000,
) -> list[dict]:
    """Returns raw reference records for `paper_id` (an S2 paperId, or
    "DOI:<doi>" / "ARXIV:<id>"), each with the S2-annotated `contexts`
    (citation-context sentences) and the referenced paper's title/externalIds.
    Used by eval/build_eval_set.py to build grounded (sentence, cited-paper)
    pairs without needing our own PDF/citation parsing."""
    url = _REFERENCES_URL_TMPL.format(paper_id=paper_id)
    params = {"fields": _REFERENCES_FIELDS, "limit": limit}
    headers = {"x-api-key": api_key} if api_key else {}

    await _throttle()

    async def call():
        return await client.get(url, params=params, headers=headers, timeout=20)

    response = await with_backoff(call)
    if response.status_code != 200:
        return []
    return response.json().get("data", [])


async def get_citations(
    client: httpx.AsyncClient,
    paper_id: str,
    *,
    api_key: str | None = None,
    limit: int = 1000,
) -> list[dict]:
    """Returns raw records for papers that CITE `paper_id` (forward citation
    search — the reverse of get_references). Used for literature-gap checks:
    "who has cited this key paper" is a more reliable way to find related
    work than keyword search alone (snowballing)."""
    url = _CITATIONS_URL_TMPL.format(paper_id=paper_id)
    params = {"fields": _CITATIONS_FIELDS, "limit": limit}
    headers = {"x-api-key": api_key} if api_key else {}

    await _throttle()

    async def call():
        return await client.get(url, params=params, headers=headers, timeout=20)

    response = await with_backoff(call)
    if response.status_code != 200:
        return []
    return response.json().get("data", [])

import asyncio
import time
import xml.etree.ElementTree as ET

import httpx

from refsearch.models import Paper
from refsearch.sources.retry import with_backoff

_API_URL = "http://export.arxiv.org/api/query"
_MIN_DELAY = 3.0
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

_last_call_lock = asyncio.Lock()
_last_call_time = 0.0


async def _throttle():
    global _last_call_time
    async with _last_call_lock:
        elapsed = time.monotonic() - _last_call_time
        if elapsed < _MIN_DELAY:
            await asyncio.sleep(_MIN_DELAY - elapsed)
        _last_call_time = time.monotonic()


async def search(client: httpx.AsyncClient, query: str, *, limit: int = 20) -> list[Paper]:
    await _throttle()
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": limit,
    }

    async def call():
        return await client.get(_API_URL, params=params, timeout=20)

    response = await with_backoff(call)
    if response.status_code != 200:
        return []

    root = ET.fromstring(response.text)
    papers = []
    for entry in root.findall("atom:entry", _NS):
        title = (entry.findtext("atom:title", default="", namespaces=_NS) or "").strip()
        abstract = (entry.findtext("atom:summary", default="", namespaces=_NS) or "").strip()
        authors = [
            (a.findtext("atom:name", default="", namespaces=_NS) or "").strip()
            for a in entry.findall("atom:author", _NS)
        ]
        published = entry.findtext("atom:published", default="", namespaces=_NS) or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        arxiv_id_url = entry.findtext("atom:id", default="", namespaces=_NS) or ""
        arxiv_id = arxiv_id_url.rstrip("/").split("/")[-1]
        doi = entry.findtext("arxiv:doi", default=None, namespaces=_NS)
        primary_category_el = entry.find("arxiv:primary_category", _NS)
        primary_category = (
            primary_category_el.get("term") if primary_category_el is not None else ""
        )
        papers.append(
            Paper(
                title=title,
                authors=authors,
                year=year,
                venue="arXiv",
                doi=doi,
                abstract=abstract,
                url=arxiv_id_url,
                source="arxiv",
                arxiv_id=arxiv_id,
                fields_of_study=[primary_category] if primary_category else [],
            )
        )
    return papers

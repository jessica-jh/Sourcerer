import asyncio

import httpx

from refsearch.models import Paper
from refsearch.sources import openalex

GRAPH_EXPAND_TOP_N = 3
GRAPH_EXPAND_MAX_PAPERS = 40
_REFERENCED_WORKS_PER_CANDIDATE = 20


async def expand_via_citation_graph(
    client: httpx.AsyncClient, top_candidates: list[Paper], *, api_key: str | None = None
) -> list[Paper]:
    """CiteAgent-inspired signal: given the current best-scored candidates,
    pull their OpenAlex reference list and forward-citation list and surface
    those neighboring papers as additional search candidates. A paper genuinely
    being cited by the claim sentence is often one hop away from a loosely-related
    paper the keyword/embedding search already found, even when direct keyword
    search can't reach it.

    Runs on OpenAlex rather than Semantic Scholar: S2's citation-graph endpoints
    (get_references/get_citations) share the same ~1req/s per-key quota as S2
    search, and this expansion step can add many extra calls per requery round --
    in practice that quota was observed to exhaust within a single eval session.
    OpenAlex's rate limit is far more generous, at the cost of one extra title-
    search call per candidate to resolve it to an OpenAlex work id (arXiv papers
    are often indexed under a different DOI than the one arXiv issues, so DOI
    matching isn't reliable here -- see openalex.find_work_by_title)."""
    if not api_key:
        return []

    candidates = top_candidates[:GRAPH_EXPAND_TOP_N]
    works = await asyncio.gather(
        *(openalex.find_work_by_title(client, c.title, api_key=api_key) for c in candidates),
        return_exceptions=True,
    )
    works = [w for w in works if isinstance(w, dict)]
    if not works:
        return []

    referenced_ids: list[str] = []
    for work in works:
        referenced_ids.extend((work.get("referenced_works") or [])[:_REFERENCED_WORKS_PER_CANDIDATE])

    tasks = [openalex.get_works_by_ids(client, referenced_ids, api_key=api_key, limit=GRAPH_EXPAND_MAX_PAPERS)]
    for work in works:
        tasks.append(openalex.get_citing_works(client, work["id"], api_key=api_key, limit=20))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    papers: list[Paper] = []
    for result in results:
        if isinstance(result, Exception):
            continue
        for item in result:
            papers.append(openalex.paper_from_work(item))
            if len(papers) >= GRAPH_EXPAND_MAX_PAPERS:
                return papers
    return papers

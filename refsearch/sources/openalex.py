import httpx

from refsearch.models import Paper
from refsearch.sources.retry import with_backoff

_API_URL = "https://api.openalex.org/works"


class OpenAlexKeyMissing(RuntimeError):
    pass


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions.append((idx, word))
    positions.sort(key=lambda p: p[0])
    return " ".join(word for _, word in positions)


def paper_from_work(item: dict) -> Paper:
    authorships = item.get("authorships") or []
    primary_location = item.get("primary_location") or {}
    source = primary_location.get("source") or {}
    return Paper(
        title=item.get("title") or item.get("display_name") or "",
        authors=[(a.get("author") or {}).get("display_name", "") for a in authorships],
        year=item.get("publication_year"),
        venue=source.get("display_name") or "",
        doi=(item.get("doi") or "").replace("https://doi.org/", "") or None,
        abstract=_reconstruct_abstract(item.get("abstract_inverted_index")),
        url=item.get("id") or "",
        source="openalex",
        fields_of_study=[c.get("display_name", "") for c in (item.get("concepts") or [])],
    )


async def search(
    client: httpx.AsyncClient,
    query: str,
    *,
    api_key: str,
    limit: int = 20,
) -> list[Paper]:
    if not api_key:
        raise OpenAlexKeyMissing(
            "OpenAlex requires an API key (--openalex-api-key or OPENALEX_API_KEY)."
        )
    params = {"search": query, "per_page": limit, "api_key": api_key}

    async def call():
        return await client.get(_API_URL, params=params, timeout=20)

    response = await with_backoff(call)
    if response.status_code != 200:
        return []
    data = response.json()
    return [paper_from_work(item) for item in data.get("results", [])]


async def find_work_by_title(
    client: httpx.AsyncClient, title: str, *, api_key: str, similarity_threshold: float = 0.9
) -> dict | None:
    """Resolves an arbitrary paper (from any source) to its OpenAlex work record
    by title search + similarity match, returning the raw work dict (which already
    includes `referenced_works`, needed by get_referenced_works/get_citing_works
    below). Title search is used rather than DOI/arXiv-id lookup because OpenAlex
    frequently indexes arXiv preprints under a different (e.g. journal-assigned)
    DOI than the arXiv-issued one, making DOI matching unreliable for this corpus."""
    from difflib import SequenceMatcher

    if not api_key or not title:
        return None
    params = {"search": title, "per_page": 3, "api_key": api_key}

    async def call():
        return await client.get(_API_URL, params=params, timeout=20)

    response = await with_backoff(call)
    if response.status_code != 200:
        return None
    for item in response.json().get("results", []):
        candidate_title = item.get("title") or item.get("display_name") or ""
        if SequenceMatcher(None, title.lower(), candidate_title.lower()).ratio() >= similarity_threshold:
            return item
    return None


async def get_works_by_ids(
    client: httpx.AsyncClient, work_ids: list[str], *, api_key: str, limit: int = 50
) -> list[dict]:
    """Batch-fetches full work records for OpenAlex work ids (short form, e.g.
    "W123..." or the full "https://openalex.org/W123..." url -- both accepted by
    the filter). Used to resolve a work's `referenced_works` id list into full
    records (title/abstract/etc)."""
    if not work_ids:
        return []
    short_ids = [wid.rsplit("/", 1)[-1] for wid in work_ids[:limit]]
    params = {"filter": "ids.openalex:" + "|".join(short_ids), "per_page": limit, "api_key": api_key}

    async def call():
        return await client.get(_API_URL, params=params, timeout=20)

    response = await with_backoff(call)
    if response.status_code != 200:
        return []
    return response.json().get("results", [])


async def get_citing_works(
    client: httpx.AsyncClient, work_id: str, *, api_key: str, limit: int = 50
) -> list[dict]:
    """Forward-citation search: papers that cite `work_id` (short or full form).
    Unlike get_works_by_ids, this filter returns full records directly (no
    separate id list to resolve)."""
    if not work_id:
        return []
    short_id = work_id.rsplit("/", 1)[-1]
    params = {"filter": f"cites:{short_id}", "per_page": limit, "api_key": api_key}

    async def call():
        return await client.get(_API_URL, params=params, timeout=20)

    response = await with_backoff(call)
    if response.status_code != 200:
        return []
    return response.json().get("results", [])

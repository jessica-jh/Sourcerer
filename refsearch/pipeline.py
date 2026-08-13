import asyncio
import json
import sys
from dataclasses import dataclass

import httpx

from refsearch import query_gen
from refsearch.citation import format_apa, format_bibtex
from refsearch.dedupe import merge_candidates
from refsearch.models import Paper, ScoredPaper
from refsearch.scoring import claude as claude_scoring
from refsearch.scoring import embedding as embedding_scoring
from refsearch.scoring import graph_expand
from refsearch.scoring import keyword as keyword_scoring
from refsearch.scoring import rerank as rerank_scoring
from refsearch.sources import arxiv, openalex, semantic_scholar
from refsearch.sources.openalex import OpenAlexKeyMissing
from refsearch.venues import filter_by_preset

DEFAULT_MODEL = claude_scoring.DEFAULT_MODEL


@dataclass
class RunConfig:
    sentence: str
    method: str = "embedding"
    venue_preset: str = "none"
    include_workshops: bool = False
    field: str | None = None
    top: int = 5
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    s2_api_key: str | None = None
    openalex_api_key: str | None = None
    # None means "auto": on when method is claude/all (an OpenAI key is
    # already required there), off otherwise. Pass True/False to override
    # independently of the scoring method -- e.g. embedding-only runs can
    # opt into either without switching to claude scoring.
    use_hyde: bool | None = None
    use_llm_query: bool | None = None
    # Cross-encoder re-scores the bi-encoder's top RERANK_CANDIDATES before
    # they're used as the "embedding" method's results or the claude
    # shortlist. Off by default: it's a local-compute cost (no API), but not
    # tied to any method the way hyde/llm_query are tied to claude.
    use_rerank: bool = False
    # Both below are independent of use_hyde/use_llm_query/use_rerank and of each
    # other -- kept as separate flags so all combinations can be ablated in eval.
    # Citation-graph expansion: pulls OpenAlex references/citations of the current
    # best candidates as extra search candidates during a requery round (CiteAgent-
    # style signal; uses OpenAlex rather than S2 to avoid S2's tight per-key rate
    # limit). Only has an effect when the requery loop runs, i.e. method claude/all.
    use_citation_graph: bool = False
    # Attribution verification: re-judges the claude shortlist for genuine citation
    # vs. mere topical relation/re-citation (CiteGuard-style check), downweighting
    # topical matches before the loop's stop/continue decision is made.
    use_verify: bool = False


RERANK_CANDIDATES = 30


async def _gather_candidates(
    client: httpx.AsyncClient, queries: list[str], config: RunConfig
) -> list[Paper]:
    tasks = []
    for q in queries:
        tasks.append(semantic_scholar.search(client, q, api_key=config.s2_api_key, field=config.field))
        tasks.append(arxiv.search(client, q))
        if config.openalex_api_key:
            tasks.append(openalex.search(client, q, api_key=config.openalex_api_key))
        elif not queries.index(q):
            print(
                "warning: no OpenAlex API key provided (--openalex-api-key / "
                "OPENALEX_API_KEY) — skipping OpenAlex source.",
                file=sys.stderr,
            )

    results = await asyncio.gather(*tasks, return_exceptions=True)
    papers: list[Paper] = []
    for result in results:
        if isinstance(result, Exception):
            if not isinstance(result, OpenAlexKeyMissing):
                print(f"warning: source search failed: {result}", file=sys.stderr)
            continue
        papers.extend(result)
    return papers


def _score(method: str, sentence: str, papers: list[Paper], hyde_text: str | None = None) -> list[ScoredPaper]:
    if method == "keyword":
        return keyword_scoring.score(sentence, papers)
    if method == "embedding":
        return embedding_scoring.score(sentence, papers, hyde_text=hyde_text)
    raise ValueError(f"unknown local method: {method}")


async def run(config: RunConfig) -> dict[str, list[ScoredPaper]]:
    queries = query_gen.base_queries(config.sentence)
    hyde_text = None

    needs_claude = config.method in ("claude", "all")
    use_hyde = needs_claude if config.use_hyde is None else config.use_hyde
    use_llm_query = needs_claude if config.use_llm_query is None else config.use_llm_query

    if (needs_claude or use_hyde or use_llm_query) and not config.api_key:
        raise ValueError(
            "--api-key (or OPENAI_API_KEY) is required for claude/all methods, "
            "or when --use-hyde / --use-llm-query is enabled"
        )
    if use_llm_query:
        llm_query = await claude_scoring.extract_search_query(
            config.sentence, api_key=config.api_key, model=config.model
        )
        if llm_query:
            queries.append(llm_query)
    if use_hyde:
        hyde_text = await claude_scoring.generate_hyde(config.sentence, api_key=config.api_key, model=config.model)
        queries.append(query_gen.clean_query(hyde_text))

    async with httpx.AsyncClient(follow_redirects=True) as client:
        papers = await _gather_candidates(client, queries, config)

        rounds = 0
        results_by_method: dict[str, list[ScoredPaper]] = {}
        while True:
            papers = merge_candidates(papers)
            papers = filter_by_preset(papers, config.venue_preset, include_workshops=config.include_workshops)

            results_by_method = {}
            if config.method in ("keyword", "all"):
                results_by_method["keyword"] = _score("keyword", config.sentence, papers)

            embedding_unsorted = None
            ranked = None
            if config.method in ("embedding", "all") or needs_claude:
                embedding_unsorted = embedding_scoring.score_unsorted(
                    config.sentence, papers, hyde_text=hyde_text
                )
                ranked = sorted(embedding_unsorted, key=lambda sp: sp.score, reverse=True)
                if config.use_rerank:
                    ranked = rerank_scoring.rerank(config.sentence, ranked, top_k=RERANK_CANDIDATES)
            if config.method in ("embedding", "all"):
                results_by_method["embedding"] = ranked
            if needs_claude:
                # Per spec §5.0 rule 5: the evidence sentence shown with the claude
                # rationale must come from the embedding extraction (§5.1), not the LLM.
                evidence_lookup = {
                    id(sp.paper): (sp.evidence_sentence, sp.overlapping_keywords, sp.evidence_sentences)
                    for sp in embedding_unsorted
                }
                shortlist = ranked[: claude_scoring.MAX_CANDIDATES]
                results_by_method["claude"] = await claude_scoring.score(
                    config.sentence,
                    [sp.paper for sp in shortlist],
                    api_key=config.api_key,
                    model=config.model,
                    evidence_lookup=evidence_lookup,
                )
                if config.use_verify:
                    results_by_method["claude"] = await claude_scoring.verify_attribution(
                        config.sentence,
                        results_by_method["claude"],
                        api_key=config.api_key,
                        model=config.model,
                    )

            if not needs_claude or rounds >= claude_scoring.MAX_RESEARCH_ROUNDS:
                break
            claude_results = results_by_method["claude"]
            top_score = claude_results[0].score if claude_results else 0.0
            if top_score >= claude_scoring.RESEARCH_THRESHOLD:
                break

            requery = await claude_scoring.suggest_requery(
                config.sentence, claude_results, api_key=config.api_key, model=config.model
            )
            new_papers = await _gather_candidates(client, [requery], config)
            if config.use_citation_graph:
                new_papers = new_papers + await graph_expand.expand_via_citation_graph(
                    client, [sp.paper for sp in claude_results], api_key=config.openalex_api_key
                )
            papers = papers + new_papers
            rounds += 1

    return results_by_method


def save_results_jsonl(results_by_method: dict[str, list[ScoredPaper]], path: str) -> None:
    with open(path, "a") as f:
        for method, scored_papers in results_by_method.items():
            for sp in scored_papers:
                record = {
                    "method": method,
                    "score": sp.score,
                    "title": sp.paper.title,
                    "authors": sp.paper.authors,
                    "year": sp.paper.year,
                    "venue": sp.paper.venue,
                    "doi": sp.paper.doi,
                    "url": sp.paper.url,
                    "source": sp.paper.source,
                    "evidence_sentence": sp.evidence_sentence,
                    "overlapping_keywords": sp.overlapping_keywords,
                    "rationale": sp.rationale,
                    "apa": format_apa(sp.paper),
                }
                f.write(json.dumps(record) + "\n")


def append_bibtex(paper: Paper, path: str) -> None:
    with open(path, "a") as f:
        f.write(format_bibtex(paper) + "\n\n")

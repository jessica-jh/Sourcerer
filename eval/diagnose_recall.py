"""Diagnoses whether FTPR/claude misses are a retrieval problem (gold paper never
in the candidate pool) or a ranking problem (gold paper was retrieved but scored
too low to make top-30 shortlist / top-5 final). Reuses the same search+dedupe+
filter+HyDE steps as refsearch.pipeline.run(method="claude"), but skips the LLM
judge call and instead reports the gold paper's rank in the full embedding-sorted
candidate pool.
"""

import argparse
import asyncio
import json
import os
import sys

import httpx
from dotenv import load_dotenv

from eval.run_eval import _gold_titles, _is_gold_match
from refsearch import query_gen
from refsearch.dedupe import merge_candidates
from refsearch.pipeline import RERANK_CANDIDATES, RunConfig, _gather_candidates
from refsearch.scoring import claude as claude_scoring
from refsearch.scoring import embedding as embedding_scoring
from refsearch.scoring import rerank as rerank_scoring
from refsearch.venues import filter_by_preset

load_dotenv()


async def diagnose_one(client: httpx.AsyncClient, example: dict, config: RunConfig) -> dict:
    queries = query_gen.base_queries(config.sentence)
    hyde_text = None
    if config.use_llm_query:
        llm_query = await claude_scoring.extract_search_query(
            config.sentence, api_key=config.api_key, model=config.model
        )
        if llm_query:
            queries.append(llm_query)
    if config.use_hyde:
        hyde_text = await claude_scoring.generate_hyde(
            config.sentence, api_key=config.api_key, model=config.model
        )
        queries.append(query_gen.clean_query(hyde_text))

    papers = await _gather_candidates(client, queries, config)
    papers = merge_candidates(papers)
    papers = filter_by_preset(papers, config.venue_preset, include_workshops=config.include_workshops)

    ranked = sorted(
        embedding_scoring.score_unsorted(config.sentence, papers, hyde_text=hyde_text),
        key=lambda sp: sp.score,
        reverse=True,
    )
    if config.use_rerank:
        ranked = rerank_scoring.rerank(config.sentence, ranked, top_k=RERANK_CANDIDATES)

    rank = None
    for i, sp in enumerate(ranked, start=1):
        if _is_gold_match(sp.paper, example):
            rank = i
            break

    return {
        "gold_title": _gold_titles(example)[0][:60],
        "pool_size": len(ranked),
        "rank": rank,
        "in_top5": rank is not None and rank <= 5,
        "in_top30": rank is not None and rank <= claude_scoring.MAX_CANDIDATES,
    }


async def main_async(
    limit: int,
    api_key: str,
    s2_api_key: str | None,
    openalex_api_key: str | None,
    use_hyde: bool,
    use_llm_query: bool,
    use_rerank: bool,
):
    with open("eval/cs_ai_eval.jsonl") as f:
        examples = [json.loads(line) for line in f][:limit]

    config = RunConfig(
        sentence="",
        method="claude",
        api_key=api_key,
        s2_api_key=s2_api_key,
        openalex_api_key=openalex_api_key,
        use_hyde=use_hyde,
        use_llm_query=use_llm_query,
        use_rerank=use_rerank,
    )

    results = []
    async with httpx.AsyncClient(follow_redirects=True) as client:
        for i, example in enumerate(examples, start=1):
            config.sentence = example["sentence"]
            try:
                r = await diagnose_one(client, example, config)
            except Exception as exc:
                print(f"[{i}/{len(examples)}] ERROR: {exc}", file=sys.stderr)
                continue
            results.append(r)
            found = "not found" if r["rank"] is None else f"rank {r['rank']}/{r['pool_size']}"
            print(f"[{i}/{len(examples)}] {found}  gold={r['gold_title']!r}", file=sys.stderr)

    never_found = [r for r in results if r["rank"] is None]
    found_not_top30 = [r for r in results if r["rank"] is not None and not r["in_top30"]]
    found_not_top5 = [r for r in results if r["in_top30"] and not r["in_top5"]]
    found_top5 = [r for r in results if r["in_top5"]]

    n = len(results)
    print(f"\n=== recall diagnosis (n={n}) ===")
    print(f"never in candidate pool (retrieval miss):      {len(never_found)}/{n}")
    print(f"in pool but outside top-{claude_scoring.MAX_CANDIDATES} shortlist:      {len(found_not_top30)}/{n}")
    print(f"in top-{claude_scoring.MAX_CANDIDATES} shortlist but outside top-5:      {len(found_not_top5)}/{n}")
    print(f"in top-5 by embedding score alone:              {len(found_top5)}/{n}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--s2-api-key", default=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))
    parser.add_argument("--openalex-api-key", default=os.environ.get("OPENALEX_API_KEY"))
    parser.add_argument("--use-hyde", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-llm-query", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-rerank", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(
        args.limit, args.api_key, args.s2_api_key, args.openalex_api_key,
        args.use_hyde, args.use_llm_query, args.use_rerank,
    ))


if __name__ == "__main__":
    main()

"""Runs reference_finder's pipeline over an eval set built by build_eval_set.py
or build_eval_set_citeme.py, and reports Recall@5 / HR@5 (reference_finder_eval_spec.md
§3). We only check "is any gold paper in the top-K" (a hit/miss judgment), so
the two metrics are computed identically here even for CiteME examples with
multiple valid gold papers -- both are reported for consistency with the spec.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from dotenv import load_dotenv

from refsearch.pipeline import RunConfig, run

load_dotenv()

TITLE_MATCH_THRESHOLD = 0.9
_VERSION_SUFFIX_RE = re.compile(r"v\d+$")


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _strip_version(arxiv_id: str) -> str:
    return _VERSION_SUFFIX_RE.sub("", arxiv_id)


_ARXIV_URL_ID_RE = re.compile(r"arxiv\.org/(?:pdf|abs)/(\d{4}\.\d{4,5})")


def _gold_titles(example: dict) -> list[str]:
    """Supports both eval-set schemas: FTPR's single gold_title, and CiteME's
    gold_titles list (some CiteME excerpts have >1 valid target paper)."""
    if "gold_titles" in example:
        return example["gold_titles"]
    return [example["gold_title"]]


def _gold_arxiv_ids(example: dict) -> list[str]:
    ids = []
    if example.get("gold_arxiv_id"):
        ids.append(example["gold_arxiv_id"])
    for url in example.get("gold_urls") or []:
        m = _ARXIV_URL_ID_RE.search(url or "")
        if m:
            ids.append(m.group(1))
    return ids


def _is_gold_match(paper, example: dict) -> bool:
    gold_arxiv_ids = _gold_arxiv_ids(example)
    if paper.arxiv_id and _strip_version(paper.arxiv_id) in gold_arxiv_ids:
        return True
    if paper.doi and any(gid in paper.doi for gid in gold_arxiv_ids):
        return True
    return any(
        _title_similarity(paper.title, gold_title) >= TITLE_MATCH_THRESHOLD
        for gold_title in _gold_titles(example)
    )


@dataclass
class EvalStats:
    total: int = 0
    hits_at_5: int = 0
    misses: list[dict] = field(default_factory=list)

    @property
    def recall_at_5(self) -> float:
        return self.hits_at_5 / self.total if self.total else 0.0

    @property
    def hr_at_5(self) -> float:
        return self.recall_at_5  # single-gold-item setup: identical by construction


async def run_eval(
    examples: list[dict],
    *,
    method: str,
    venue_preset: str,
    top: int,
    api_key: str | None,
    s2_api_key: str | None,
    openalex_api_key: str | None,
    use_hyde: bool | None = None,
    use_llm_query: bool | None = None,
    use_rerank: bool = False,
    use_citation_graph: bool = False,
    use_verify: bool = False,
) -> EvalStats:
    stats = EvalStats()
    for i, example in enumerate(examples, start=1):
        config = RunConfig(
            sentence=example["sentence"],
            method=method,
            venue_preset=venue_preset,
            top=top,
            api_key=api_key,
            s2_api_key=s2_api_key,
            openalex_api_key=openalex_api_key,
            use_hyde=use_hyde,
            use_llm_query=use_llm_query,
            use_rerank=use_rerank,
            use_citation_graph=use_citation_graph,
            use_verify=use_verify,
        )
        try:
            results_by_method = await run(config)
        except Exception as exc:
            print(f"[{i}/{len(examples)}] ERROR: {exc}", file=sys.stderr)
            stats.total += 1
            stats.misses.append({**example, "error": str(exc)})
            continue

        scored = results_by_method.get(method, [])
        top_k = scored[:top]
        hit = any(_is_gold_match(sp.paper, example) for sp in top_k)

        stats.total += 1
        if hit:
            stats.hits_at_5 += 1
        else:
            stats.misses.append(example)

        status = "HIT" if hit else "miss"
        print(f"[{i}/{len(examples)}] {status}  gold={_gold_titles(example)[0][:60]!r}", file=sys.stderr)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default="eval/cs_ai_eval.jsonl")
    parser.add_argument("--method", choices=["keyword", "embedding", "claude"], default="embedding")
    parser.add_argument("--venue-preset", choices=["business", "cs_ai", "psych_soc", "none"], default="none")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--limit", type=int, default=20, help="Number of eval examples to run (default 20).")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--s2-api-key", default=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))
    parser.add_argument("--openalex-api-key", default=os.environ.get("OPENALEX_API_KEY"))
    parser.add_argument("--use-hyde", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-llm-query", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-rerank", action="store_true")
    parser.add_argument("--use-citation-graph", action="store_true")
    parser.add_argument("--use-verify", action="store_true")
    args = parser.parse_args()

    with open(args.eval_set) as f:
        examples = [json.loads(line) for line in f]
    examples = examples[: args.limit]

    stats = asyncio.run(
        run_eval(
            examples,
            method=args.method,
            venue_preset=args.venue_preset,
            top=args.top,
            api_key=args.api_key,
            s2_api_key=args.s2_api_key,
            openalex_api_key=args.openalex_api_key,
            use_hyde=args.use_hyde,
            use_llm_query=args.use_llm_query,
            use_rerank=args.use_rerank,
            use_citation_graph=args.use_citation_graph,
            use_verify=args.use_verify,
        )
    )

    print(f"\n=== {args.method} / venue-preset={args.venue_preset} / n={stats.total} ===")
    print(f"Recall@{args.top}: {stats.recall_at_5:.3f}")
    print(f"HR@{args.top}:     {stats.hr_at_5:.3f}")
    if stats.misses:
        print(f"\n{len(stats.misses)} misses (first 5):")
        for m in stats.misses[:5]:
            label = m.get("context_id") or m.get("citeme_id") or "?"
            print(f"  - {_gold_titles(m)[0][:70]!r}  (id={label})")


if __name__ == "__main__":
    main()

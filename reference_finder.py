import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

from refsearch.citation import format_apa
from refsearch.pipeline import RunConfig, append_bibtex, run, save_results_jsonl

load_dotenv()

RESULTS_PATH = "results.jsonl"
BIBTEX_PATH = "references.bib"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find grounded academic references for a claim sentence.")
    parser.add_argument("sentence", help="The English academic sentence needing a citation.")
    parser.add_argument("--method", choices=["keyword", "embedding", "claude", "all"], default="embedding")
    parser.add_argument("--venue-preset", choices=["business", "cs_ai", "psych_soc", "none"], default="none")
    parser.add_argument("--include-workshops", action="store_true")
    parser.add_argument("--field", default=None)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--s2-api-key", default=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))
    parser.add_argument("--openalex-api-key", default=os.environ.get("OPENALEX_API_KEY"))
    parser.add_argument(
        "--use-hyde", action=argparse.BooleanOptionalAction, default=None,
        help="Add a HyDE-generated hypothetical abstract as an extra search query. "
             "Default: on for --method claude/all, off otherwise.",
    )
    parser.add_argument(
        "--use-llm-query", action=argparse.BooleanOptionalAction, default=None,
        help="Add an LLM-distilled short search query (fixes long/unfocused queries "
             "from long claim passages). Default: on for --method claude/all, off otherwise.",
    )
    parser.add_argument(
        "--use-rerank", action="store_true",
        help="Re-score the top candidates with a cross-encoder (local, no API cost) "
             "instead of relying on bi-encoder cosine similarity alone.",
    )
    parser.add_argument(
        "--use-citation-graph", action="store_true",
        help="During a requery round, also pull OpenAlex references/citations of "
             "the current best candidates as extra search candidates. Requires "
             "--openalex-api-key. Only has an effect for --method claude/all "
             "(the requery loop only runs there).",
    )
    parser.add_argument(
        "--use-verify", action="store_true",
        help="Re-judge the claude shortlist for genuine citation attribution vs. "
             "mere topical relation/re-citation, downweighting topical matches.",
    )
    return parser.parse_args(argv)


def print_results(method: str, scored_papers, top: int) -> None:
    print(f"\n=== {method} ===")
    if not scored_papers:
        print("No candidates found.")
        return
    for rank, sp in enumerate(scored_papers[:top], start=1):
        print(f"\n[{rank}] score={sp.score:.3f}  source={sp.paper.source}")
        print(f"    {format_apa(sp.paper)}")
        if sp.evidence_sentence:
            print(f"    Evidence: {sp.evidence_sentence}")
        if sp.overlapping_keywords:
            print(f"    Overlapping keywords: {', '.join(sp.overlapping_keywords)}")
        if sp.rationale:
            print(f"    Rationale: {sp.rationale}")


def main() -> None:
    args = parse_args(sys.argv[1:])

    needs_key = args.method in ("claude", "all") or args.use_hyde or args.use_llm_query or args.use_verify
    if needs_key and not args.api_key:
        print(
            "error: --api-key or OPENAI_API_KEY is required for --method claude/all, "
            "or when --use-hyde / --use-llm-query / --use-verify is enabled",
            file=sys.stderr,
        )
        sys.exit(1)

    config = RunConfig(
        sentence=args.sentence,
        method=args.method,
        venue_preset=args.venue_preset,
        include_workshops=args.include_workshops,
        field=args.field,
        top=args.top,
        api_key=args.api_key,
        model=args.model,
        s2_api_key=args.s2_api_key,
        openalex_api_key=args.openalex_api_key,
        use_hyde=args.use_hyde,
        use_llm_query=args.use_llm_query,
        use_rerank=args.use_rerank,
        use_citation_graph=args.use_citation_graph,
        use_verify=args.use_verify,
    )

    results_by_method = asyncio.run(run(config))

    for method, scored_papers in results_by_method.items():
        print_results(method, scored_papers, args.top)

    save_results_jsonl(results_by_method, RESULTS_PATH)

    all_scored = [sp for scored in results_by_method.values() for sp in scored[: args.top]]
    if not all_scored:
        return
    print(f"\nSaved candidate details to {RESULTS_PATH}.")
    choice = input(f"\nEnter a rank number (1-{min(len(all_scored), args.top)}) to append its BibTeX to "
                    f"{BIBTEX_PATH}, or press Enter to skip: ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(all_scored):
            append_bibtex(all_scored[idx].paper, BIBTEX_PATH)
            print(f"Appended to {BIBTEX_PATH}.")


if __name__ == "__main__":
    main()

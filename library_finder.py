import argparse
import asyncio
import os
import sys
from difflib import SequenceMatcher

from dotenv import load_dotenv

from refsearch import library
from refsearch.citation import format_apa
from refsearch.library_pipeline import find_supporting, verify_citation

load_dotenv()

TITLE_MATCH_THRESHOLD = 0.5


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def cmd_add(args: argparse.Namespace) -> None:
    paper = asyncio.run(library.ingest_pdf(args.pdf_path, grobid_url=args.grobid_url))
    print(f"Added: {format_apa(paper)}")
    print(f"  abstract: {len(paper.abstract)} chars, full text: {len(paper.full_text)} chars")


def cmd_list(_args: argparse.Namespace) -> None:
    papers = library.load_library()
    if not papers:
        print("Library is empty. Add papers with: library_finder.py add <pdf_path>")
        return
    for i, paper in enumerate(papers, start=1):
        print(f"[{i}] {format_apa(paper)}")


def cmd_search(args: argparse.Namespace) -> None:
    papers = library.load_library()
    if not papers:
        print("Library is empty. Add papers with: library_finder.py add <pdf_path>")
        return
    results = asyncio.run(find_supporting(args.sentence, papers, api_key=args.api_key, model=args.model))
    if not results:
        print("No candidates found.")
        return
    for rank, sp in enumerate(results[: args.top], start=1):
        print(f"\n[{rank}] score={sp.score:.3f}")
        print(f"    {format_apa(sp.paper)}")
        if sp.evidence_sentence:
            print(f"    Evidence: {sp.evidence_sentence}")
        if sp.rationale:
            print(f"    Rationale: {sp.rationale}")


def _matches_title(paper_title: str, query: str) -> bool:
    if query.lower() in paper_title.lower():
        return True
    return _title_similarity(paper_title, query) >= TITLE_MATCH_THRESHOLD


def cmd_verify(args: argparse.Namespace) -> None:
    papers = library.load_library()
    matches = [p for p in papers if _matches_title(p.title, args.paper)]
    if not matches:
        print(f"No library paper matching title {args.paper!r}. Try `library_finder.py list` to see titles.")
        return
    matches.sort(key=lambda p: _title_similarity(p.title, args.paper), reverse=True)
    paper = matches[0]
    print(f"Matched: {format_apa(paper)}")
    result = asyncio.run(verify_citation(args.sentence, paper, api_key=args.api_key, model=args.model))
    print(f"\nscore={result.score:.3f}")
    if result.evidence_sentence:
        print(f"Evidence: {result.evidence_sentence}")
    if result.rationale:
        print(f"Rationale: {result.rationale}")


def cmd_rebuild_embeddings(_args: argparse.Namespace) -> None:
    count = library.rebuild_embeddings()
    print(f"Rebuilt embeddings for {count} paper(s) from library/index.jsonl.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Personal PDF library citation search/verification.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--model", default="gpt-4o-mini")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Ingest a PDF into the library via GROBID.")
    add_parser.add_argument("pdf_path")
    add_parser.add_argument("--grobid-url", default="http://localhost:8070")
    add_parser.set_defaults(func=cmd_add)

    list_parser = subparsers.add_parser("list", help="List papers in the library.")
    list_parser.set_defaults(func=cmd_list)

    search_parser = subparsers.add_parser("search", help="Find library papers supporting a claim sentence.")
    search_parser.add_argument("sentence")
    search_parser.add_argument("--top", type=int, default=5)
    search_parser.set_defaults(func=cmd_search)

    verify_parser = subparsers.add_parser("verify", help="Check whether citing a specific paper here is appropriate.")
    verify_parser.add_argument("sentence")
    verify_parser.add_argument("--paper", required=True, help="Title (or a substring of it) of a library paper.")
    verify_parser.set_defaults(func=cmd_verify)

    rebuild_parser = subparsers.add_parser(
        "rebuild-embeddings",
        help="Recompute the embedding cache for all library papers from index.jsonl (recovers a deleted/corrupted cache).",
    )
    rebuild_parser.set_defaults(func=cmd_rebuild_embeddings)

    args = parser.parse_args(sys.argv[1:])

    if not args.api_key:
        print("error: --api-key or OPENAI_API_KEY is required.", file=sys.stderr)
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()

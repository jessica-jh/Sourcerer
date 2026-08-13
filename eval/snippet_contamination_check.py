"""Cheap pre-check for the re-citation/misattribution risk in S2's
snippet-search API (used by search_text_snippet in CiteGuard, and the action
we're considering porting into refsearch). No PDF access involved -- this
hits https://api.semanticscholar.org/graph/v1/snippet/search directly.

For N CiteME excerpts, fetches the top snippet matches and writes them to a
CSV with a blank `is_own_claim` column for manual tagging: does the returned
snippet text represent the matched paper's OWN claim, or is it itself
quoting/attributing the idea to a different (third-party) source? Run
eval/snippet_contamination_report.py afterward to tally the tagged results.
"""

import argparse
import csv
import os
import re
import sys

import httpx
from dotenv import load_dotenv

from eval.build_eval_set_citeme import _clean_excerpt

load_dotenv()

SNIPPET_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/snippet/search"
_DOI_URL_RE = re.compile(r"https://doi\.org/\S+")


def _paper_url(paper: dict) -> str:
    disclaimer = (paper.get("openAccessInfo") or {}).get("disclaimer", "")
    m = _DOI_URL_RE.search(disclaimer)
    if m:
        return m.group(0).rstrip(".,")
    corpus_id = paper.get("corpusId")
    if corpus_id:
        return f"https://www.semanticscholar.org/paper/{corpus_id}"
    return ""


def fetch_snippets(client: httpx.Client, query: str, api_key: str | None, limit: int = 5) -> list[dict]:
    headers = {"x-api-key": api_key} if api_key else {}
    response = client.get(
        SNIPPET_SEARCH_URL,
        params={"query": query, "limit": limit, "fieldsOfStudy": "Computer Science"},
        headers=headers,
        timeout=20,
    )
    if response.status_code != 200:
        print(f"  warning: {response.status_code} for query {query[:60]!r}", file=sys.stderr)
        return []
    return response.json().get("data", [])


def build(n: int, output_path: str, s2_api_key: str | None) -> None:
    from datasets import load_dataset

    ds = load_dataset("bethgelab/CiteME")["train"]
    ds = ds.filter(lambda r: r["split"] == "test")
    rows = list(ds)[:n]

    out_rows = []
    with httpx.Client(follow_redirects=True) as client:
        for i, r in enumerate(rows, start=1):
            sentence = _clean_excerpt(r["excerpt"])
            gold_title = r["target_paper_title"].split("[TITLE_SEPARATOR]")[0]
            snippets = fetch_snippets(client, sentence, s2_api_key, limit=1)
            print(f"[{i}/{len(rows)}] {len(snippets)} snippets for citeme_id={r['id']}", file=sys.stderr)
            if not snippets:
                out_rows.append(
                    {
                        "citeme_id": r["id"],
                        "excerpt": r["excerpt"],
                        "gold_title": gold_title,
                        "rank": "",
                        "snippet_paper_title": "(no snippets found)",
                        "snippet_paper_url": "",
                        "snippet_section": "",
                        "snippet_text": "",
                        "is_core_claim": "",
                        "note": "",
                    }
                )
                continue
            for rank, item in enumerate(snippets, start=1):
                snippet = item.get("snippet", {})
                paper = item.get("paper", {})
                out_rows.append(
                    {
                        "citeme_id": r["id"],
                        "excerpt": r["excerpt"],
                        "gold_title": gold_title,
                        "rank": rank,
                        "snippet_paper_title": paper.get("title", ""),
                        "snippet_paper_url": _paper_url(paper),
                        "snippet_section": snippet.get("section", ""),
                        "snippet_text": snippet.get("text", ""),
                        "is_core_claim": "",  # fill in by hand: yes / no / unsure
                        "note": "",
                    }
                )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "citeme_id",
                "excerpt",
                "gold_title",
                "rank",
                "snippet_paper_title",
                "snippet_paper_url",
                "snippet_section",
                "snippet_text",
                "is_core_claim",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nwrote {len(out_rows)} rows ({len(rows)} excerpts) to {output_path}", file=sys.stderr)
    print(
        "Open 'snippet_paper_url' and check the source directly, then fill in "
        "'is_core_claim' by hand: yes / no / unsure -- is this the paper's own "
        "central claim, or just a passing mention / something attributed to a "
        "third party within that paper?",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20, help="Number of CiteME test excerpts to check.")
    parser.add_argument("--output", default="eval/snippet_review.csv")
    parser.add_argument("--s2-api-key", default=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))
    args = parser.parse_args()
    build(args.n, args.output, args.s2_api_key)


if __name__ == "__main__":
    main()

"""Builds eval/cs_ai_eval.jsonl from the FullTextPeerRead (FTPR) Local Citation
Recommendation dataset (Gu et al., ECIR 2022 -- see
https://github.com/nianlonggu/Local-Citation-Recommendation), per
reference_finder_eval_spec.md §2.2. FTPR is the smallest of the five standard
LCR datasets and is recommended there as the first pilot.

Expects eval/data/peerread/{contexts.json,papers.json,test.json}, downloaded
from the repo's Google Drive link (see README pointer in that repo).

Per spec: citation markers are removed to produce plain, citation-free
sentences; contexts citing more than one gold paper (positive_ids) are
excluded (our tool's design assumes one supporting paper per sentence);
sentences shorter than MIN_WORDS are excluded.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

MIN_WORDS = 10
MARKER_RE = re.compile(r"TARGETCIT|OTHERCIT")
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}v\d+$")

DATA_DIR = Path(__file__).parent / "data" / "peerread"


def _clean_sentence(masked_text: str) -> str:
    stripped = MARKER_RE.sub("", masked_text)
    return re.sub(r"\s+", " ", stripped).strip()


def build(target: int, output_path: str, seed: int) -> None:
    with open(DATA_DIR / "contexts.json") as f:
        contexts = json.load(f)
    with open(DATA_DIR / "papers.json") as f:
        papers = json.load(f)
    with open(DATA_DIR / "test.json") as f:
        test = json.load(f)

    random.Random(seed).shuffle(test)

    examples = []
    skipped_bundled = 0
    skipped_short = 0
    skipped_missing = 0
    for entry in test:
        if len(examples) >= target:
            break
        positive_ids = entry["positive_ids"]
        if len(positive_ids) != 1:
            skipped_bundled += 1
            continue
        context = contexts.get(entry["context_id"])
        gold_paper = papers.get(positive_ids[0])
        if not context or not gold_paper or not gold_paper.get("title"):
            skipped_missing += 1
            continue

        sentence = _clean_sentence(context["masked_text"])
        if len(sentence.split()) < MIN_WORDS:
            skipped_short += 1
            continue

        raw_id = positive_ids[0]
        gold_arxiv_id = re.sub(r"v\d+$", "", raw_id) if _ARXIV_ID_RE.match(raw_id) else None
        examples.append(
            {
                "sentence": sentence,
                "gold_title": gold_paper["title"],
                "gold_arxiv_id": gold_arxiv_id,
                "gold_year": gold_paper.get("year"),
                "gold_venue": gold_paper.get("venue"),
                "context_id": entry["context_id"],
            }
        )

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(
        f"wrote {len(examples)} examples to {output_path} "
        f"(skipped: {skipped_bundled} bundled, {skipped_short} too short, {skipped_missing} missing data)",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--output", default="eval/cs_ai_eval.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(args.target, args.output, args.seed)


if __name__ == "__main__":
    main()

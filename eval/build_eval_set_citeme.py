"""Builds eval/cs_ai_eval_citeme.jsonl from the CiteME benchmark (Wingate/
Bethge lab, NeurIPS 2024 D&B track -- https://github.com/bethgelab/CiteME).

CiteME excerpts are human-curated (unlike the automatically-extracted, noisy
FullTextPeerRead contexts in build_eval_set.py), each with a single [CITATION]
marker. A minority of excerpts have more than one valid target paper (joined
by "[TITLE_SEPARATOR]" in target_paper_title, e.g. commonly co-cited papers)
-- run_eval.py must treat a hit against ANY of them as correct.

Uses the `datasets` library, which pulled the gated HF mirror without needing
an explicit token in testing (the repo grants access broadly once you request
it on huggingface.co).
"""

import argparse
import json
import re
import sys

MARKER_RE = re.compile(r"\[CITATION\]")


def _clean_excerpt(excerpt: str) -> str:
    return re.sub(r"\s+", " ", MARKER_RE.sub("", excerpt)).strip()


def build(output_path: str, split: str) -> None:
    from datasets import load_dataset

    ds = load_dataset("bethgelab/CiteME")["train"]
    if split != "all":
        ds = ds.filter(lambda r: r["split"] == split)

    examples = []
    for row in ds:
        gold_titles = [t.strip() for t in row["target_paper_title"].split("[TITLE_SEPARATOR]")]
        examples.append(
            {
                "sentence": _clean_excerpt(row["excerpt"]),
                "gold_titles": gold_titles,
                "gold_urls": [u.strip() for u in row["target_paper_url"].split("[TITLE_SEPARATOR]")]
                if "[TITLE_SEPARATOR]" in row["target_paper_url"]
                else [row["target_paper_url"]],
                "source_paper_title": row["source_paper_title"],
                "year": row["year"],
                "citeme_id": row["id"],
            }
        )

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"wrote {len(examples)} examples to {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="eval/cs_ai_eval_citeme.jsonl")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    args = parser.parse_args()
    build(args.output, args.split)


if __name__ == "__main__":
    main()

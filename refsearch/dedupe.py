from difflib import SequenceMatcher

from refsearch.models import Paper

TITLE_SIMILARITY_THRESHOLD = 0.9


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def merge_candidates(papers: list[Paper]) -> list[Paper]:
    merged: list[Paper] = []
    doi_index: dict[str, int] = {}

    for paper in papers:
        doi_key = paper.dedupe_key_doi()
        if doi_key and doi_key in doi_index:
            continue
        duplicate_idx = None
        for idx, existing in enumerate(merged):
            if _title_similarity(paper.title, existing.title) >= TITLE_SIMILARITY_THRESHOLD:
                duplicate_idx = idx
                break
        if duplicate_idx is not None:
            existing = merged[duplicate_idx]
            if not existing.abstract and paper.abstract:
                merged[duplicate_idx] = paper
                if doi_key:
                    doi_index[doi_key] = duplicate_idx
            continue
        merged.append(paper)
        if doi_key:
            doi_index[doi_key] = len(merged) - 1

    return merged

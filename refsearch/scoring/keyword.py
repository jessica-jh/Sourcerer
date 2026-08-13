import re

from refsearch.models import Paper, ScoredPaper

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")
_STOPWORDS = {
    "a", "an", "the", "this", "that", "of", "in", "on", "at", "to", "for",
    "with", "by", "as", "and", "or", "but", "is", "are", "was", "were",
    "it", "its", "their", "our", "we", "they",
}


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS}


def score(sentence: str, papers: list[Paper]) -> list[ScoredPaper]:
    query_words = _words(sentence)
    scored = []
    for paper in papers:
        abstract_words = _words(paper.abstract or paper.title)
        overlap = query_words & abstract_words
        union = query_words | abstract_words
        jaccard = len(overlap) / len(union) if union else 0.0
        scored.append(
            ScoredPaper(
                paper=paper,
                score=jaccard,
                evidence_sentence=(paper.abstract or paper.title)[:280],
                overlapping_keywords=sorted(overlap),
            )
        )
    scored.sort(key=lambda sp: sp.score, reverse=True)
    return scored

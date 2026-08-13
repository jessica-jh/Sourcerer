from refsearch.models import ScoredPaper

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank(sentence: str, scored_papers: list[ScoredPaper], top_k: int | None = None) -> list[ScoredPaper]:
    """Re-scores the top `top_k` (or all, if None) bi-encoder results with a
    cross-encoder that attends to the claim and candidate text jointly, rather
    than comparing independently-computed embeddings. This targets the false
    positives the bi-encoder (refsearch.scoring.embedding) is prone to: a
    candidate sentence that shares vocabulary with the claim but isn't
    actually relevant can still score high on cosine similarity alone.
    Candidates beyond top_k are appended unchanged, keeping their bi-encoder
    rank/score, so the output is still a full list over all input papers."""
    if not scored_papers:
        return []
    head = scored_papers[:top_k] if top_k else scored_papers
    tail = scored_papers[len(head):]

    model = _get_model()
    pairs = [(sentence, sp.evidence_sentence or sp.paper.abstract or sp.paper.title) for sp in head]
    cross_scores = model.predict(pairs)

    reranked_head = [
        ScoredPaper(
            paper=sp.paper,
            score=float(cross_score),
            evidence_sentence=sp.evidence_sentence,
            overlapping_keywords=sp.overlapping_keywords,
            rationale=sp.rationale,
        )
        for sp, cross_score in zip(head, cross_scores)
    ]
    reranked_head.sort(key=lambda sp: sp.score, reverse=True)
    return reranked_head + tail

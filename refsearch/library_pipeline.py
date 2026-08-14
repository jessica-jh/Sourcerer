from refsearch.models import Paper, ScoredPaper
from refsearch.scoring import claude as claude_scoring
from refsearch.scoring import embedding as embedding_scoring
from refsearch.scoring import rerank as rerank_scoring


async def find_supporting(
    sentence: str,
    library: list[Paper],
    *,
    api_key: str,
    model: str = claude_scoring.DEFAULT_MODEL,
) -> list[ScoredPaper]:
    """Ranks the local library against `sentence`, same scoring as the live
    pipeline (embedding pre-rank -> cross-encoder rerank -> claude judge on a
    shortlist) but with no external search, dedupe, or venue filtering -- the
    library is already the full, curated candidate set.

    The rerank step matters more here than it might seem: the bi-encoder
    (embedding pre-rank) embeds the claim and each candidate sentence
    independently, so it can't tell a candidate that merely shares vocabulary
    with the claim from one that's actually relevant. The cross-encoder
    attends to the claim and candidate jointly instead, which catches that
    class of false positive before it ever reaches the claude judge."""
    if not library:
        return []
    embedding_unsorted = embedding_scoring.score_unsorted(sentence, library)
    ranked = sorted(embedding_unsorted, key=lambda sp: sp.score, reverse=True)
    ranked = rerank_scoring.rerank(sentence, ranked, top_k=claude_scoring.MAX_CANDIDATES)
    evidence_lookup = {
        id(sp.paper): (sp.evidence_sentence, sp.overlapping_keywords, sp.evidence_sentences)
        for sp in embedding_unsorted
    }
    shortlist = ranked[: claude_scoring.MAX_CANDIDATES]
    return await claude_scoring.score(
        sentence,
        [sp.paper for sp in shortlist],
        api_key=api_key,
        model=model,
        evidence_lookup=evidence_lookup,
    )


async def verify_citation(
    sentence: str,
    paper: Paper,
    *,
    api_key: str,
    model: str = claude_scoring.DEFAULT_MODEL,
) -> ScoredPaper:
    """Answers "does citing `paper` here make sense?" for a single (sentence,
    paper) pair, reusing claude_scoring.verify_attribution -- built for
    re-judging a shortlist, but works the same on a list of one."""
    [scored] = embedding_scoring.score_unsorted(sentence, [paper])
    [verified] = await claude_scoring.verify_attribution(sentence, [scored], api_key=api_key, model=model)
    return verified

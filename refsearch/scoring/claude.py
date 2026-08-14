import json
import sys

from refsearch.models import Paper, ScoredPaper

RESEARCH_THRESHOLD = 0.5
MAX_RESEARCH_ROUNDS = 2
DEFAULT_MODEL = "gpt-4o-mini"

# Above this many candidates in one prompt, gpt-4o-mini starts skipping some
# and hallucinating extra ids to pad out the enumeration (observed: ids like
# "arxiv:083" for a call with far fewer real candidates). Cap what we ask it
# to judge in one shot; embedding score (already computed for evidence
# extraction) picks the shortlist.
MAX_CANDIDATES = 30

# The LLM is never shown a slot to emit bibliographic text (title/authors/venue) —
# its only output channel is: pick a candidate id, a 1-5 relevance score, and a
# rationale. Citation strings and evidence sentences are assembled/extracted by
# code from the real candidate data (see refsearch/citation.py and
# refsearch/scoring/embedding.py), never typed by the model.
_JUDGE_SYSTEM = """You judge whether candidate academic papers actually support a claim \
sentence from a research paper. You must only reason about the candidates provided — \
never invent papers, and never output a paper's title, authors, or venue. For each \
candidate id, output a relevance score from 1 (irrelevant) to 5 (directly and strongly \
supports the claim) and a one-sentence rationale."""


def _candidate_id(idx: int) -> str:
    return str(idx)


_CONTEXT_WINDOW_CHARS = 500


def _context_window(full_text: str, evidence_sentences: list[str], window_chars: int = _CONTEXT_WINDOW_CHARS) -> str:
    """Expands the top-K matching sentences (refsearch.scoring.embedding picks
    more than one for full_text papers, see TOP_K_EVIDENCE_SENTENCES) into
    surrounding windows of body text, merging overlapping windows. A claim can
    combine facts spread across sentences that aren't adjacent (e.g. "dense
    Transformer" stated in one section, "405B parameters" in another) that a
    single best-match sentence -- or even its immediate neighbors -- wouldn't
    contain, causing the judge to see a technically-true but incomplete excerpt
    and misjudge it as unsupported."""
    if not evidence_sentences or not any(evidence_sentences):
        return full_text[:800]

    spans: list[tuple[int, int]] = []
    for sentence in evidence_sentences:
        if not sentence:
            continue
        pos = full_text.find(sentence)
        if pos == -1:
            continue
        spans.append((max(0, pos - window_chars), min(len(full_text), pos + len(sentence) + window_chars)))

    if not spans:
        return evidence_sentences[0]

    spans.sort()
    merged: list[list[int]] = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return "\n[...]\n".join(full_text[start:end] for start, end in merged)


def _build_prompt(
    sentence: str,
    papers: list[Paper],
    candidate_ids: list[str],
    evidence_lookup: dict[int, tuple[str, list[str], list[str]]] | None = None,
) -> str:
    evidence_lookup = evidence_lookup or {}
    lines = [f'Claim sentence: "{sentence}"', "", "Candidates:"]
    for cand_id, paper in zip(candidate_ids, papers):
        if paper.full_text:
            # Full-text (library) candidates: show a window of body text around the
            # sentence the embedding step located, not an arbitrary prefix of the
            # text -- unlike an abstract, the first ~800 chars of a paper's body are
            # rarely where a specific claim is actually discussed. A window (not just
            # the lone sentence) lets the judge see facts split across adjacent
            # sentences that a single sentence would miss.
            _, _, evidence_sentences = evidence_lookup.get(id(paper), ("", [], []))
            text = _context_window(paper.full_text, evidence_sentences)
        else:
            text = (paper.abstract or "")[:800]
        lines.append(f"[{cand_id}] Abstract: {text or '(no abstract available)'}")
    lines.append(
        "\nRespond ONLY with a JSON array, one object per candidate id, each with "
        'keys "id" (must exactly match one of the bracketed candidate ids above), '
        '"score" (1-5), and "rationale" (<=10 words, referring only to the abstract '
        "content, never to a title/author/venue you were not given)."
    )
    return "\n".join(lines)


async def score(
    sentence: str,
    papers: list[Paper],
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    evidence_lookup: dict[int, tuple[str, list[str], list[str]]] | None = None,
) -> list[ScoredPaper]:
    """evidence_lookup maps id(paper) -> (evidence_sentence, overlapping_keywords),
    typically produced by refsearch.scoring.embedding.score_unsorted. Per spec §5.0
    rule 5, the evidence sentence shown alongside the claude rationale must come from
    that extraction step, never from the LLM itself."""
    if not papers:
        return []
    evidence_lookup = evidence_lookup or {}

    candidate_ids = [_candidate_id(idx) for idx in range(len(papers))]
    id_to_paper = dict(zip(candidate_ids, papers))

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    prompt = _build_prompt(sentence, papers, candidate_ids, evidence_lookup)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    text = response.choices[0].message.content or ""
    try:
        judgments = json.loads(text[text.index("[") : text.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        judgments = []

    by_id: dict[str, dict] = {}
    for j in judgments:
        if not isinstance(j, dict):
            continue
        cand_id = str(j.get("id"))
        if cand_id not in id_to_paper:
            print(
                f"warning: claude scorer returned unknown candidate id {cand_id!r}; discarding.",
                file=sys.stderr,
            )
            continue
        by_id[cand_id] = j

    scored = []
    for cand_id, paper in zip(candidate_ids, papers):
        judgment = by_id.get(cand_id, {})
        raw_score = judgment.get("score", 0)
        evidence_sentence, overlapping_keywords, evidence_sentences = evidence_lookup.get(id(paper), ("", [], []))
        scored.append(
            ScoredPaper(
                paper=paper,
                score=float(raw_score) / 5.0,
                evidence_sentence=evidence_sentence,
                overlapping_keywords=overlapping_keywords,
                rationale=judgment.get("rationale", ""),
                evidence_sentences=evidence_sentences,
            )
        )
    scored.sort(key=lambda sp: sp.score, reverse=True)
    return scored


async def extract_search_query(sentence: str, *, api_key: str, model: str = DEFAULT_MODEL) -> str:
    """Distills a long citation-context passage (often 50-100+ words, see
    eval/build_eval_set.py's masked_text) down to a short, focused search
    query. The naive alternative (refsearch.query_gen.base_queries, which
    strips stopwords but keeps everything else, or picks the longest N words)
    dilutes the paper's actual topic among incidental words and was found to
    be the dominant cause of retrieval misses (diagnosed via
    eval/diagnose_recall.py: gold papers were absent from the candidate pool
    entirely, not merely ranked low)."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract a short academic search query (3-5 keywords, never more) "
                    "that captures the single main topic, method, or contribution being "
                    "described in the passage below, suitable for a paper search engine "
                    "like Semantic Scholar. These engines return few or zero results once "
                    "queries get long, so prefer the smallest set of terms that still "
                    "identifies the core idea over an exhaustive list. Output ONLY the "
                    "keywords separated by spaces — no punctuation, no explanation, no quotes."
                ),
            },
            {"role": "user", "content": sentence},
        ],
    )
    return (response.choices[0].message.content or "").strip()


async def generate_hyde(sentence: str, *, api_key: str, model: str = DEFAULT_MODEL) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Write a single hypothetical academic abstract paragraph (no title, "
                    "no citation) that would plausibly support the given claim sentence, "
                    "in the style of a real paper abstract."
                ),
            },
            {"role": "user", "content": sentence},
        ],
    )
    return (response.choices[0].message.content or "").strip()


_VERIFY_SYSTEM = """You check whether a candidate paper is the GENUINE original source of a claim \
sentence, merely topically related to it, or actually CONTRADICTS it. Classify as one of three \
attributions:
- "genuine": the candidate's abstract itself describes the specific finding, method, or result \
that the claim sentence attributes to it.
- "topical": the abstract is about a broader or adjacent subject, or reads like a paper that \
discusses/relates to/builds on the claim's topic without being the paper that produced the \
specific result described — including the case where the claim is actually citing a survey or \
follow-up work that itself cites the true original source.
- "contradicts": the abstract directly states something that conflicts with, disproves, or is \
inconsistent with what the claim sentence says — citing this candidate here would misrepresent \
what it actually found. Use this even when the candidate is otherwise highly relevant to the \
topic; relevance does not imply support.
For each candidate id, output one of these three labels and a one-sentence rationale, reasoning \
only from the abstract content provided."""


def _build_verify_prompt(
    sentence: str,
    papers: list[Paper],
    candidate_ids: list[str],
    evidence_lookup: dict[int, tuple[str, list[str], list[str]]] | None = None,
) -> str:
    evidence_lookup = evidence_lookup or {}
    lines = [f'Claim sentence: "{sentence}"', "", "Candidates:"]
    for cand_id, paper in zip(candidate_ids, papers):
        if paper.full_text:
            _, _, evidence_sentences = evidence_lookup.get(id(paper), ("", [], []))
            text = _context_window(paper.full_text, evidence_sentences)
        else:
            text = (paper.abstract or "")[:800]
        lines.append(f"[{cand_id}] Abstract: {text or '(no abstract available)'}")
    lines.append(
        "\nRespond ONLY with a JSON array, one object per candidate id, each with "
        'keys "id" (must exactly match one of the bracketed candidate ids above), '
        '"attribution" ("genuine", "topical", or "contradicts"), and "rationale" (<=25 words).'
    )
    return "\n".join(lines)


async def verify_attribution(
    sentence: str,
    scored_papers: list[ScoredPaper],
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> list[ScoredPaper]:
    """CiteGuard-inspired check (see project memory citeguard-side-investigation):
    snippet/abstract-based search frequently mistakes a paper that merely discusses
    or re-cites the claim's topic for the paper that actually produced the cited
    result. This re-judges the claude shortlist specifically for that failure mode
    and downweights (not discards, since the check itself can be wrong) candidates
    classified as "topical" rather than "genuine" attribution."""
    if not scored_papers:
        return []

    candidate_ids = [_candidate_id(idx) for idx in range(len(scored_papers))]
    papers = [sp.paper for sp in scored_papers]
    id_to_scored = dict(zip(candidate_ids, scored_papers))
    evidence_lookup = {
        id(sp.paper): (sp.evidence_sentence, sp.overlapping_keywords, sp.evidence_sentences) for sp in scored_papers
    }

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    prompt = _build_verify_prompt(sentence, papers, candidate_ids, evidence_lookup)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _VERIFY_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    text = response.choices[0].message.content or ""
    try:
        judgments = json.loads(text[text.index("[") : text.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        judgments = []

    verdict_by_id: dict[str, dict] = {}
    for j in judgments:
        if not isinstance(j, dict):
            continue
        cand_id = str(j.get("id"))
        if cand_id in id_to_scored:
            verdict_by_id[cand_id] = j

    adjusted = []
    for cand_id, sp in zip(candidate_ids, scored_papers):
        verdict = verdict_by_id.get(cand_id, {})
        attribution = verdict.get("attribution", "genuine")
        factor = {"topical": 0.5, "contradicts": 0.0}.get(attribution, 1.0)
        verify_rationale = verdict.get("rationale", "")
        rationale = f"[{attribution}] {verify_rationale}" if verify_rationale else sp.rationale
        adjusted.append(
            ScoredPaper(
                paper=sp.paper,
                score=sp.score * factor,
                evidence_sentence=sp.evidence_sentence,
                overlapping_keywords=sp.overlapping_keywords,
                rationale=rationale,
            )
        )
    adjusted.sort(key=lambda sp: sp.score, reverse=True)
    return adjusted


async def suggest_requery(
    sentence: str, top_results: list[ScoredPaper], *, api_key: str, model: str = DEFAULT_MODEL
) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    summary = "\n".join(f"- score {sp.score:.2f}: {sp.evidence_sentence or '(no evidence)'}" for sp in top_results[:5])
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "The current top search results are weak matches for the claim sentence. "
                    "Suggest a better, more specific search query (terms only, no explanation) "
                    "that might retrieve stronger supporting papers."
                ),
            },
            {"role": "user", "content": f'Claim: "{sentence}"\n\nCurrent result evidence:\n{summary}'},
        ],
    )
    return (response.choices[0].message.content or "").strip()

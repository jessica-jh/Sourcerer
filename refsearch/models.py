from dataclasses import dataclass, field


@dataclass
class Paper:
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str | None = None
    abstract: str = ""
    url: str = ""
    source: str = ""
    arxiv_id: str | None = None
    fields_of_study: list[str] = field(default_factory=list)
    # Full body text (GROBID-extracted paragraphs, joined). Only populated for
    # refsearch.library papers -- live-search sources leave this empty, which
    # keeps their scoring behavior (abstract-only) unchanged.
    full_text: str = ""
    # refsearch.library-only bookkeeping: when the paper was ingested (ISO date)
    # and the filename it's stored under in library/pdfs/ (original upload name,
    # deduplicated -- not a temp/random name).
    added_at: str = ""
    pdf_filename: str = ""
    # Single-folder classification (a paper belongs to exactly one project/topic
    # at a time) so search/verify can be scoped to one research thread instead
    # of the whole library.
    collection: str = "Uncategorized"

    def dedupe_key_doi(self) -> str | None:
        return self.doi.strip().lower() if self.doi else None

    def dedupe_key_title(self) -> str:
        return "".join(ch.lower() for ch in self.title if ch.isalnum())


@dataclass
class ScoredPaper:
    paper: Paper
    score: float
    evidence_sentence: str = ""
    overlapping_keywords: list[str] = field(default_factory=list)
    rationale: str = ""
    # Top-K matching sentences (evidence_sentence is evidence_sentences[0]).
    # Populated with more than one entry only for full_text (library) papers,
    # where a claim can be split across adjacent sentences that a single best
    # match would miss -- see refsearch.scoring.embedding.score_unsorted.
    evidence_sentences: list[str] = field(default_factory=list)

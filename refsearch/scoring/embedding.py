import os
import re

import numpy as np

from refsearch.models import Paper, ScoredPaper
from refsearch.scoring.keyword import _words

_MODEL_NAME = "all-MiniLM-L6-v2"
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

_model = None

# Library papers can run to hundreds of sentences of full_text; re-encoding
# all of them on every search/verify call (refsearch/library_pipeline.py)
# made those noticeably slow. Precomputed at ingest/reparse time
# (refsearch.library.ingest_pdf/reparse_pdf call build_and_cache below) and
# reused here instead of re-embedding on every query.
EMBEDDING_CACHE_DIR = os.path.join("library", "embeddings")


def _cache_path(pdf_filename: str) -> str:
    return os.path.join(EMBEDDING_CACHE_DIR, f"{pdf_filename}.npz")


def _searchable_text(paper: Paper) -> str:
    return f"{paper.abstract} {paper.full_text}".strip() if paper.full_text else paper.abstract


def build_and_cache(paper: Paper) -> None:
    """Computes and stores sentence embeddings for a library paper's searchable
    text. No-op for papers without a pdf_filename (live-search results) or
    with no text to embed."""
    if not paper.pdf_filename:
        return
    sentences = split_sentences(_searchable_text(paper)) or [paper.title]
    vectors = _get_model().encode(sentences)
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    np.savez(_cache_path(paper.pdf_filename), sentences=np.array(sentences, dtype=object), vectors=vectors)


def delete_cache(pdf_filename: str) -> None:
    path = _cache_path(pdf_filename)
    if os.path.exists(path):
        os.remove(path)


def _load_cache(pdf_filename: str) -> tuple[list[str], np.ndarray] | None:
    path = _cache_path(pdf_filename)
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        return list(data["sentences"]), data["vectors"]
    except Exception:
        return None

# For full_text (library) papers, a claim can draw on facts split across
# adjacent-but-distinct sentences (e.g. one names the architecture, another
# states the parameter count) that the single best-matching sentence would
# miss. Surfacing the top few candidates -- not just the top one -- gives the
# claude judge (refsearch.scoring.claude) a real chance of seeing both.
TOP_K_EVIDENCE_SENTENCES = 3


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = _SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(query_vec: np.ndarray, sent_vecs: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    sent_norms = sent_vecs / (np.linalg.norm(sent_vecs, axis=1, keepdims=True) + 1e-9)
    return sent_norms @ query_norm


def score_unsorted(sentence: str, papers: list[Paper], hyde_text: str | None = None) -> list[ScoredPaper]:
    """Same as score(), but preserves input order — used when other scorers
    (e.g. claude) need to align evidence sentences back to specific papers by position."""
    model = _get_model()
    query_text = hyde_text if hyde_text else sentence
    query_vec = model.encode(query_text)
    query_words = _words(sentence)

    scored = []
    for paper in papers:
        cached = _load_cache(paper.pdf_filename) if paper.pdf_filename else None
        if cached:
            sentences, sent_vecs = cached
        else:
            # Search abstract + body together (not full_text alone): GROBID's
            # body extraction excludes the abstract, and a claim is sometimes
            # stated more concisely there than anywhere in the body.
            sentences = split_sentences(_searchable_text(paper)) or [paper.title]
            sent_vecs = model.encode(sentences)
        sims = _cosine_similarity(query_vec, np.atleast_2d(sent_vecs))
        top_k = TOP_K_EVIDENCE_SENTENCES if paper.full_text else 1
        ranked_idx = np.argsort(sims)[::-1][:top_k]
        best_score = float(sims[ranked_idx[0]])
        evidence_sentences = [sentences[i] for i in ranked_idx]
        evidence = evidence_sentences[0]
        overlap = sorted(query_words & _words(evidence))
        scored.append(
            ScoredPaper(
                paper=paper,
                score=best_score,
                evidence_sentence=evidence,
                overlapping_keywords=overlap,
                evidence_sentences=evidence_sentences,
            )
        )
    return scored


def score(sentence: str, papers: list[Paper], hyde_text: str | None = None) -> list[ScoredPaper]:
    scored = score_unsorted(sentence, papers, hyde_text=hyde_text)
    scored.sort(key=lambda sp: sp.score, reverse=True)
    return scored

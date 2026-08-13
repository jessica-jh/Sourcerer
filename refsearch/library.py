import asyncio
import dataclasses
import datetime
import json
import os
import re
import shutil
import subprocess

import httpx

from refsearch import grobid_client
from refsearch.models import Paper
from refsearch.scoring import embedding as embedding_scoring

LIBRARY_DIR = "library"
INDEX_PATH = os.path.join(LIBRARY_DIR, "index.jsonl")
PDFS_DIR = os.path.join(LIBRARY_DIR, "pdfs")

_DEDUPE_SUFFIX_RE = re.compile(r"\s*\(\d+\)$")


def _git_commit(message: str) -> None:
    """Best-effort auto-commit of library/pdfs + index.jsonl so every ingest
    is individually recoverable from git history. Silently no-ops if this
    checkout isn't a git repo (or git isn't installed) rather than failing
    the ingest -- version control is a safety net, not a hard dependency."""
    try:
        subprocess.run(["git", "add", PDFS_DIR, INDEX_PATH], check=True, capture_output=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
            return  # nothing staged (e.g. not a git repo, or a no-op call)
        result = subprocess.run(["git", "commit", "-m", message], capture_output=True)
        if result.returncode != 0:
            print(f"warning: git auto-commit failed: {result.stdout.decode().strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"warning: git auto-commit skipped: {e}")


def load_library() -> list[Paper]:
    if not os.path.exists(INDEX_PATH):
        return []
    papers = []
    with open(INDEX_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(Paper(**json.loads(line)))
    return papers


def _unique_dest_name(filename: str) -> str:
    """Avoids silently overwriting an existing file with the same name (two
    different PDFs can share a filename, e.g. both named "paper.pdf")."""
    base, ext = os.path.splitext(filename)
    candidate = filename
    n = 1
    while os.path.exists(os.path.join(PDFS_DIR, candidate)):
        candidate = f"{base} ({n}){ext}"
        n += 1
    return candidate


class DuplicatePaper(Exception):
    def __init__(self, existing: Paper):
        self.existing = existing
        super().__init__(f"Already in library (as {existing.collection!r}): {existing.title!r}")


_DUPLICATE_TITLE_SIMILARITY = 0.9

# Concurrent ingests (refsearch.library_pipeline batch uploads use
# asyncio.gather) each read load_library() for the duplicate check before any
# of them has appended -- without serializing check+append, two concurrent
# uploads of the same PDF both see an empty/no-match result and both get
# added. Only the check+append needs to be atomic; the slow GROBID network
# call happens before acquiring this, so parallelism there is unaffected.
_ingest_lock = asyncio.Lock()


def find_duplicate(new_paper: Paper) -> Paper | None:
    from difflib import SequenceMatcher

    for p in load_library():
        if new_paper.doi and p.doi and new_paper.doi.strip().lower() == p.doi.strip().lower():
            return p
        if SequenceMatcher(None, new_paper.title.lower(), p.title.lower()).ratio() >= _DUPLICATE_TITLE_SIMILARITY:
            return p
    return None


def append_to_library(
    paper: Paper, pdf_path: str, *, original_filename: str | None = None, collection: str = "Uncategorized"
) -> None:
    os.makedirs(PDFS_DIR, exist_ok=True)
    filename = _unique_dest_name(original_filename or os.path.basename(pdf_path))
    dest = os.path.join(PDFS_DIR, filename)
    if os.path.abspath(dest) != os.path.abspath(pdf_path):
        shutil.copy2(pdf_path, dest)
    paper.pdf_filename = filename
    paper.added_at = datetime.date.today().isoformat()
    paper.collection = collection
    with open(INDEX_PATH, "a") as f:
        f.write(json.dumps(dataclasses.asdict(paper)) + "\n")
    embedding_scoring.build_and_cache(paper)
    _git_commit(f"Add to library: {paper.title!r}")


async def ingest_pdf(
    pdf_path: str,
    *,
    grobid_url: str = "http://localhost:8070",
    original_filename: str | None = None,
    collection: str = "Uncategorized",
) -> Paper:
    async with httpx.AsyncClient() as client:
        tei_xml = await grobid_client.process_pdf(client, pdf_path, grobid_url=grobid_url)
    paper = grobid_client.parse_tei(tei_xml)
    if not paper.title:
        paper.title = grobid_client.extract_metadata_title(pdf_path)
    if paper.year is None:
        paper.year = grobid_client.extract_metadata_year(pdf_path)
    if not paper.authors:
        paper.authors = grobid_client.extract_metadata_authors(pdf_path)
    if not paper.title:
        stem = os.path.splitext(original_filename or os.path.basename(pdf_path))[0]
        # Both GROBID and /Title metadata found nothing usable -- last resort
        # is the filename, but strip a "(1)"/"(2)" suffix if present so a
        # title doesn't visibly expose our own dedup-renaming scheme
        # (_unique_dest_name) to the user.
        paper.title = _DEDUPE_SUFFIX_RE.sub("", stem).strip()
    async with _ingest_lock:
        duplicate = find_duplicate(paper)
        if duplicate:
            raise DuplicatePaper(duplicate)
        append_to_library(paper, pdf_path, original_filename=original_filename, collection=collection)
    return paper


def rebuild_embeddings() -> int:
    """Recomputes the embedding cache for every paper in the library from its
    stored title/abstract/full_text. The cache is derived data (see
    embedding_scoring.build_and_cache) so this recovers from a deleted or
    corrupted library/embeddings/ directory without needing to re-run GROBID
    or re-upload PDFs. Returns the number of papers rebuilt."""
    papers = load_library()
    for paper in papers:
        embedding_scoring.build_and_cache(paper)
    return len(papers)


def list_collections() -> list[str]:
    return sorted({p.collection or "Uncategorized" for p in load_library()})


def set_collection_bulk(pdf_filenames: list[str], collection: str) -> None:
    papers = load_library()
    filenames = set(pdf_filenames)
    for p in papers:
        if p.pdf_filename in filenames:
            p.collection = collection
    _rewrite_index(papers)


def _rewrite_index(papers: list[Paper]) -> None:
    with open(INDEX_PATH, "w") as f:
        for p in papers:
            f.write(json.dumps(dataclasses.asdict(p)) + "\n")


def delete_paper(paper: Paper) -> None:
    """Removes the paper's index.jsonl record and its stored PDF (if any).
    Matches by pdf_filename when available (unambiguous), falling back to
    title for older records that predate that field."""
    papers = load_library()
    if paper.pdf_filename:
        remaining = [p for p in papers if p.pdf_filename != paper.pdf_filename]
    else:
        remaining = [p for p in papers if p.title != paper.title]
    _rewrite_index(remaining)
    if paper.pdf_filename:
        pdf_path = os.path.join(PDFS_DIR, paper.pdf_filename)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        embedding_scoring.delete_cache(paper.pdf_filename)


def update_paper(pdf_filename: str, new_paper: Paper) -> None:
    papers = load_library()
    for i, p in enumerate(papers):
        if p.pdf_filename == pdf_filename:
            papers[i] = new_paper
            break
    _rewrite_index(papers)


_TITLE_SIMILARITY_FLOOR = 0.5


async def reparse_pdf(paper: Paper, *, grobid_url: str = "http://localhost:8070") -> Paper:
    """Re-runs GROBID on the already-stored PDF (no re-upload needed) and
    updates the library record in place, keeping pdf_filename/added_at/
    collection from the existing entry. Useful after a parser fix (e.g. venue
    extraction) or just to re-check a paper without hunting down the file again.

    Merges conservatively rather than overwriting wholesale: GROBID's header
    model occasionally misfires on a given PDF (observed: picking a table
    caption as the title instead of the real one) and a naive overwrite would
    silently replace a correct field with garbage. title/year/venue/doi only
    get replaced when the new value is non-empty AND -- for title specifically
    -- close enough to the old one to plausibly be the same paper (a
    known-good title should never be swapped for an unrelated string; a
    casing/whitespace cleanup is fine, since that's near-identical text).
    abstract/authors/full_text are taken as-is when non-empty since those
    don't have the same "silently wrong but well-formed" failure mode."""
    from difflib import SequenceMatcher

    if not paper.pdf_filename:
        raise ValueError("This paper has no stored PDF file to re-parse.")
    pdf_path = os.path.join(PDFS_DIR, paper.pdf_filename)
    async with httpx.AsyncClient() as client:
        tei_xml = await grobid_client.process_pdf(client, pdf_path, grobid_url=grobid_url)
    fresh = grobid_client.parse_tei(tei_xml)
    if not fresh.title:
        fresh.title = grobid_client.extract_metadata_title(pdf_path)
    if fresh.year is None:
        fresh.year = grobid_client.extract_metadata_year(pdf_path)

    title = paper.title
    if fresh.title and (
        not paper.title
        or SequenceMatcher(None, paper.title.lower(), fresh.title.lower()).ratio() >= _TITLE_SIMILARITY_FLOOR
    ):
        title = fresh.title

    merged = Paper(
        title=title,
        authors=fresh.authors or paper.authors,
        year=fresh.year if fresh.year is not None else paper.year,
        venue=fresh.venue or paper.venue,
        doi=fresh.doi or paper.doi,
        abstract=fresh.abstract or paper.abstract,
        full_text=fresh.full_text or paper.full_text,
        source=paper.source,
        pdf_filename=paper.pdf_filename,
        added_at=paper.added_at,
        collection=paper.collection,
    )
    update_paper(paper.pdf_filename, merged)
    embedding_scoring.build_and_cache(merged)
    return merged

# Reference Library

A personal citation-finding tool: upload your PDF library, ask "which of my papers supports this sentence?", and get back a ranked, cited answer — with signals about *how much to trust each match* before you actually cite it.

Built as a Streamlit app on top of GROBID (PDF parsing), sentence-transformer embeddings, a cross-encoder reranker, and an LLM judge.

## Why

Writing a literature review means constantly asking "do I already have a source for this claim?" This tool searches your own ingested PDF library (not the open web) and returns:

- a ranked list of candidate papers with a relevance score
- the exact evidence sentence it's judging from
- an in-text citation string ready to paste into a draft (`Wen and Zhu (2019)`)
- **which section of the paper the evidence came from** — Introduction/Literature Review vs. Results/Discussion — so you can tell a paper's own finding apart from it merely citing someone else's
- a flag when the evidence sentence itself looks like it's citing a different source (re-citation risk)

## Screenshots

**Library** — upload PDFs, browse/search your collection, click a row to see details inline:

![Library tab](docs/screenshots/01_library_overview.png)
![Library row detail](docs/screenshots/02_library_row_detail.png)

**Search** — ask a claim sentence, get ranked results with provenance signals:

![Search results with section provenance and re-citation warning](docs/screenshots/03_search_provenance.png)

## Architecture

**Ingest** (uploading a PDF into the library):

![Ingest pipeline](docs/architecture-ingest.svg)

- GROBID extracts title/authors/year/venue/abstract/full text, with `consolidateHeader=1` cross-checking against CrossRef — GROBID's own layout-based parser can badly mis-segment author names or miss the venue entirely; consolidation fixes most of that for free.
- Placeholder/garbage metadata (`"untitled"`, an internal tracking code masquerading as a title) is rejected rather than trusted verbatim.
- Section headings (`Introduction`, `Results`, ...) are preserved as inline markers in the stored full text instead of being flattened away, so evidence can later be traced back to which section it came from.
- Missing venue gets a one-time OpenAlex lookup by DOI, written permanently into the record — never a per-search cost.
- Every ingest is auto-committed to git, so the library has real version history.

**Search** (asking a claim sentence):

![Search pipeline](docs/architecture-search.svg)

- **Embedding pre-rank**: a bi-encoder (`all-MiniLM-L6-v2`) compares the claim against cached per-sentence embeddings for every paper. Cheap, but it embeds the claim and each candidate independently — no cross-attention, so it can be fooled by shared vocabulary that isn't actually relevant.
- **Cross-encoder rerank**: `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores the top 30 by attending to the claim and evidence text *jointly*, catching the bi-encoder's false positives. Not cacheable (score depends on the specific claim), so it only runs on the pre-ranked shortlist.
- **LLM judge**: `gpt-4o-mini` gives a final 1–5 relevance score and a short rationale for each shortlisted candidate.
- **Provenance annotations**: computed directly from the paper's own text (no extra LLM call) — which section the evidence sentence lives in, and a regex check for an embedded citation marker (`(Smith, 2020)`, `[12]`, ...) suggesting the sentence is itself reciting a different source.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in OPENAI_API_KEY at minimum
```

GROBID runs as a local Docker container:

```bash
./start_app.command   # starts GROBID if needed, then the Streamlit app
```

or manually:

```bash
docker run -d --name grobid -p 8070:8070 grobid/grobid:0.8.1
streamlit run app.py
```

## Also included: a CLI for live external search

`reference_finder.py` searches Semantic Scholar / arXiv / OpenAlex live (not your library) for a claim sentence — separate from the Streamlit app, useful for finding papers you don't have yet:

```bash
python reference_finder.py "Platform owners entering complementary markets can crowd out third-party innovation." --method all
```

Supports HyDE query expansion, an LLM-distilled search query, citation-graph expansion, and an iterative requery loop when the top match scores low — see `--help` for the full flag list. `library_finder.py` is the equivalent CLI for library-only search/verify, if you'd rather not use the Streamlit UI.

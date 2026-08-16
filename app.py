import asyncio
import dataclasses
import os
import tempfile
import time

import httpx
import streamlit as st
from dotenv import load_dotenv

from refsearch import grobid_client, library
from refsearch.citation import format_apa, format_intext_authors, format_intext_citation, looks_like_recitation
from refsearch.library_pipeline import find_supporting, verify_citation
from refsearch.scoring import section

load_dotenv()

st.set_page_config(page_title="Reference Library", layout="wide")


@st.cache_resource
def _startup_backfill_embeddings() -> int:
    """Runs once per app process (st.cache_resource, not per-rerun) so a
    crashed/interrupted ingest's missing embedding cache (see
    library.backfill_missing_embeddings) gets repaired automatically on the
    next app start instead of silently staying slow until someone remembers
    to run `library_finder.py rebuild-embeddings` by hand."""
    return library.backfill_missing_embeddings()


_startup_backfill_embeddings()

API_KEY = os.environ.get("OPENAI_API_KEY")
GROBID_URL = "http://localhost:8070"
MODEL = "gpt-4o-mini"
NEW_COLLECTION_OPTION = "+ New collection..."
# claude judge scores are raw_score/5, so these line up with 4/5 and 2/5.
STRONG_MATCH_THRESHOLD = 0.8
WEAK_MATCH_THRESHOLD = 0.4

ATTRIBUTION_COLORS = {
    "genuine": "green",
    "topical": "orange",
    "contradicts": "red",
}

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def _attribution_from_rationale(rationale: str) -> str | None:
    for label in ATTRIBUTION_COLORS:
        if rationale.startswith(f"[{label}]"):
            return label
    return None


def _evidence_section_caption(paper, evidence_sentence: str) -> str | None:
    """Tells the user which section of `paper` the evidence sentence came
    from, and what that implies -- a Results/Discussion sentence is likely
    the paper's own claim, while an Introduction/Literature Review sentence
    is often reciting someone else's finding to set up the paper's framing.
    Returns None when the section can't be determined (abstract-only
    candidate, or a library paper ingested before section markers were
    added -- needs a re-parse via the Library tab to backfill)."""
    if not evidence_sentence or not paper.full_text:
        return None
    heading = section.find_section(paper.full_text, evidence_sentence)
    if not heading:
        return None
    category = section.classify_section(heading)
    if category == "background":
        return f"📚 From “{heading}” — may be reporting prior work, not this paper's own finding."
    if category == "own_content":
        return f"✅ From “{heading}” — appears to be this paper's own content."
    return f"From “{heading}”."


async def _check_grobid_alive() -> bool:
    async with httpx.AsyncClient() as client:
        return await grobid_client.is_alive(client, GROBID_URL)


async def _ingest_one(name: str, data: bytes, collection: str) -> tuple[str, object]:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        paper = await library.ingest_pdf(
            tmp_path, grobid_url=GROBID_URL, original_filename=name, collection=collection
        )
        return name, paper
    except Exception as exc:  # noqa: BLE001 -- surfaced per-file in the UI, not re-raised
        return name, exc
    finally:
        os.unlink(tmp_path)


async def _ingest_all(files: list[tuple[str, bytes]], collection: str) -> list[tuple[str, object]]:
    return await asyncio.gather(*(_ingest_one(name, data, collection) for name, data in files))


def _collection_picker(label: str, key: str) -> str:
    options = ["All"] + library.list_collections()
    return st.selectbox(label, options, key=key)


st.title("Reference Library")

status_col, key_col = st.columns(2)
with status_col:
    grobid_ok = asyncio.run(_check_grobid_alive())
    if grobid_ok:
        st.success(f"GROBID connected ({GROBID_URL})")
    else:
        st.error(
            f"GROBID not reachable at {GROBID_URL}. Start it with:\n\n"
            "`docker run -d --name grobid -p 8070:8070 grobid/grobid:0.8.1`"
        )
with key_col:
    if API_KEY:
        st.success("OPENAI_API_KEY set")
    else:
        st.error("OPENAI_API_KEY not set (.env) — search/verify need it.")

tab_library, tab_search, tab_verify = st.tabs(["Library", "Search", "Verify citation"])

with tab_library:
    st.subheader("Add papers")

    existing_collections = library.list_collections()
    collection_choice = st.selectbox("Collection", existing_collections + [NEW_COLLECTION_OPTION], key="add_collection")
    if collection_choice == NEW_COLLECTION_OPTION:
        collection_choice = st.text_input("New collection name", key="new_collection_name").strip() or "Uncategorized"

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type="pdf", accept_multiple_files=True, key=f"pdf_uploader_{st.session_state.uploader_key}"
    )
    if uploaded_files:
        st.caption(
            f"Estimated time: ~{len(uploaded_files) * 15}-{len(uploaded_files) * 30}s "
            f"for {len(uploaded_files)} file(s) (rough estimate, varies with paper length)"
        )
    if uploaded_files and st.button("Ingest uploaded PDFs"):
        files = [(f.name, f.getvalue()) for f in uploaded_files]
        start = time.time()
        with st.spinner(f"Parsing {len(files)} PDF(s) via GROBID (in parallel)..."):
            results = asyncio.run(_ingest_all(files, collection_choice))
        elapsed = time.time() - start
        for name, result in results:
            if isinstance(result, library.DuplicatePaper):
                st.warning(f"{name}: already in library — {result.existing.title!r} ({result.existing.collection})")
            elif isinstance(result, Exception):
                st.error(f"{name}: {result}")
            else:
                st.success(f"Added: {format_apa(result)}")
        st.caption(f"Done in {elapsed:.1f}s ({elapsed / len(files):.1f}s/paper avg)")
        st.session_state.uploader_key += 1
        st.rerun()

    st.subheader("All References")
    papers = library.load_library()
    if not papers:
        st.info("Library is empty. Upload a PDF above to get started.")
    else:
        filter_col, search_col = st.columns([1, 2])
        with filter_col:
            collection_filter = _collection_picker("Collection", key="library_collection_filter")
        with search_col:
            query = st.text_input(
                "Search",
                key="library_search",
                placeholder="Search in All References",
            )

        filtered = papers
        if collection_filter != "All":
            filtered = [p for p in filtered if p.collection == collection_filter]
        if query.strip():
            q = query.strip().lower()
            filtered = [p for p in filtered if q in p.title.lower() or q in " ".join(p.authors).lower()]
        st.caption(f"{len(filtered)} reference(s)")

        # st.dataframe's row selection turned out unreliable in practice
        # (clicking a row highlights it but doesn't always fire the
        # on_select rerun) -- back to plain st.button per cell, which
        # behaves predictably. Every cell in a row triggers the same
        # action, so clicking anywhere in the row (not just the title)
        # toggles its detail panel open/closed. CSS below strips the button
        # chrome so the columns of buttons read as table cells/rows, with a
        # bottom border per row and a hover highlight.
        st.markdown(
            """
            <style>
            div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
                text-align: left;
                justify-content: flex-start;
                border: none;
                background: transparent;
                padding: 0.35rem 0.25rem;
                border-radius: 0;
                width: 100%;
            }
            div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover {
                background: rgba(128, 128, 128, 0.15);
                color: inherit;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        select_all_col, clear_all_col = st.columns([1, 4])
        with select_all_col:
            if st.button("Select all", key="select_all_btn"):
                for p in filtered:
                    st.session_state[f"bulk_{p.pdf_filename or p.title}"] = True
        with clear_all_col:
            if st.button("Clear all", key="clear_all_btn"):
                for p in filtered:
                    st.session_state[f"bulk_{p.pdf_filename or p.title}"] = False

        row_widths = [0.6, 3.2, 2.3, 0.8, 1.5]
        header_cols = st.columns(row_widths)
        for col, label in zip(header_cols, ["", "Title", "Authors", "Year", "Collection"]):
            col.markdown(f"**{label}**")
        st.markdown('<hr style="margin: 0.1rem 0 0.4rem 0;">', unsafe_allow_html=True)

        # Detail renders inline directly under its row (accordion-style,
        # pushing later rows down) instead of in a separate side panel.
        # Page-level scroll now (not a fixed-height sub-container): with no
        # side panel to keep in view, there's no reason to cap the table's
        # height -- letting the page scroll shows more of the list at once.
        bulk_selected: list[str] = []
        for p in filtered:
            row_id = p.pdf_filename or p.title

            def _toggle_detail(pdf_filename=p.pdf_filename):
                if st.session_state.get("library_detail") == pdf_filename:
                    st.session_state.library_detail = None
                else:
                    st.session_state.library_detail = pdf_filename
                    st.session_state[f"editing_{pdf_filename}"] = False

            cols = st.columns(row_widths)
            checked = cols[0].checkbox("select", key=f"bulk_{row_id}", label_visibility="collapsed")
            if checked and p.pdf_filename:
                bulk_selected.append(p.pdf_filename)
            if cols[1].button(p.title, key=f"cell_title_{row_id}", use_container_width=True):
                _toggle_detail()
            if cols[2].button(
                format_intext_authors(p.authors), key=f"cell_authors_{row_id}", use_container_width=True
            ):
                _toggle_detail()
            if cols[3].button(str(p.year or "n.d."), key=f"cell_year_{row_id}", use_container_width=True):
                _toggle_detail()
            if cols[4].button(p.collection, key=f"cell_collection_{row_id}", use_container_width=True):
                _toggle_detail()
            st.markdown('<hr style="margin: 0.1rem 0;">', unsafe_allow_html=True)

            if p.pdf_filename and p.pdf_filename == st.session_state.get("library_detail"):
                detail_filename = p.pdf_filename
                with st.container(border=True):
                    if st.button("✕ Close", key=f"close_btn_{detail_filename}"):
                        st.session_state.library_detail = None
                        st.rerun()
                    editing = st.session_state.get(f"editing_{detail_filename}", False)
                    if not editing:
                        st.markdown(f"### {p.title}")
                        st.markdown(f"*{format_intext_authors(p.authors)}*")
                        st.markdown(
                            f"**Authors:** {', '.join(p.authors) or 'Unknown'}  \n"
                            f"**Year:** {p.year or 'n.d.'}  \n"
                            f"**Journal:** {p.venue or '—'}  \n"
                            f"**Collection:** {p.collection}"
                        )
                        if p.abstract:
                            st.markdown(f"**Abstract:** {p.abstract}")
                        else:
                            st.caption("No abstract extracted.")
                        if st.button("✏️ Edit info", key=f"edit_btn_{detail_filename}"):
                            st.session_state[f"editing_{detail_filename}"] = True
                            st.rerun()
                    else:
                        st.markdown("#### Edit paper info")
                        new_title = st.text_input("Title", value=p.title, key=f"edit_title_{detail_filename}")
                        new_authors = st.text_input(
                            "Authors (comma-separated)",
                            value=", ".join(p.authors),
                            key=f"edit_authors_{detail_filename}",
                        )
                        new_year = st.number_input(
                            "Year", value=p.year or 0, min_value=0, max_value=2100, key=f"edit_year_{detail_filename}"
                        )
                        new_venue = st.text_input("Journal", value=p.venue, key=f"edit_venue_{detail_filename}")
                        save_col, cancel_col = st.columns(2)
                        if save_col.button("Save", key=f"save_{detail_filename}"):
                            edited = dataclasses.replace(
                                p,
                                title=new_title.strip() or p.title,
                                authors=[a.strip() for a in new_authors.split(",") if a.strip()],
                                year=int(new_year) or None,
                                venue=new_venue.strip(),
                            )
                            library.update_paper(detail_filename, edited)
                            st.session_state[f"editing_{detail_filename}"] = False
                            st.rerun()
                        if cancel_col.button("Cancel", key=f"cancel_{detail_filename}"):
                            st.session_state[f"editing_{detail_filename}"] = False
                            st.rerun()

                    pdf_path = os.path.join(library.PDFS_DIR, p.pdf_filename) if p.pdf_filename else None
                    pdf_exists = bool(pdf_path and os.path.exists(pdf_path))

                    btn_col1, btn_col2, btn_col3 = st.columns(3)
                    with btn_col1:
                        if pdf_exists:
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    "Download PDF",
                                    f.read(),
                                    file_name=p.pdf_filename,
                                    mime="application/pdf",
                                    key=f"dl_{p.pdf_filename}",
                                )
                        else:
                            st.caption("Original PDF not found.")
                    with btn_col2:
                        reparse_key = f"reparse_{p.pdf_filename}"
                        if pdf_exists and st.button("🔄 Re-parse", key=reparse_key):
                            with st.spinner("Re-parsing via GROBID..."):
                                asyncio.run(library.reparse_pdf(p, grobid_url=GROBID_URL))
                            st.rerun()
                    with btn_col3:
                        delete_key = f"delete_{p.pdf_filename or p.title}"
                        confirm_key = f"confirm_{delete_key}"
                        if not st.session_state.get(confirm_key):
                            if st.button("🗑 Delete", key=delete_key):
                                st.session_state[confirm_key] = True
                                st.rerun()
                        else:
                            st.warning("Delete this paper permanently?")
                            yes_col, no_col = st.columns(2)
                            if yes_col.button("Yes, delete", key=f"{delete_key}_yes"):
                                library.delete_paper(p)
                                st.session_state.pop(confirm_key, None)
                                st.session_state.library_detail = None
                                st.rerun()
                            if no_col.button("Cancel", key=f"{delete_key}_no"):
                                st.session_state.pop(confirm_key, None)
                                st.rerun()
                st.markdown('<hr style="margin: 0.4rem 0;">', unsafe_allow_html=True)

        if bulk_selected:
            st.markdown(f"**{len(bulk_selected)} selected**")
            bulk_target = st.selectbox(
                "Move selected to collection",
                library.list_collections() + [NEW_COLLECTION_OPTION],
                key="bulk_target_collection",
                label_visibility="collapsed",
            )
            if bulk_target == NEW_COLLECTION_OPTION:
                bulk_target = st.text_input("New collection name", key="bulk_new_collection").strip() or "Uncategorized"

            move_btn_col, delete_btn_col = st.columns(2)
            with move_btn_col:
                if st.button(f"Move {len(bulk_selected)} paper(s)"):
                    library.set_collection_bulk(bulk_selected, bulk_target)
                    st.success(f"Moved to '{bulk_target}'.")
                    st.rerun()
            with delete_btn_col:
                confirm_bulk_delete = "confirm_bulk_delete"
                if not st.session_state.get(confirm_bulk_delete):
                    if st.button(f"🗑 Delete {len(bulk_selected)} paper(s)"):
                        st.session_state[confirm_bulk_delete] = True
                        st.rerun()
                else:
                    st.warning(f"Delete {len(bulk_selected)} paper(s) permanently?")
                    yes_col, no_col = st.columns(2)
                    if yes_col.button("Yes, delete", key="bulk_delete_yes"):
                        for pdf_filename in bulk_selected:
                            match = next((fp for fp in papers if fp.pdf_filename == pdf_filename), None)
                            if match:
                                library.delete_paper(match)
                        st.session_state.pop(confirm_bulk_delete, None)
                        if st.session_state.get("library_detail") in bulk_selected:
                            st.session_state.library_detail = None
                        st.rerun()
                    if no_col.button("Cancel", key="bulk_delete_no"):
                        st.session_state.pop(confirm_bulk_delete, None)
                        st.rerun()

with tab_search:
    st.subheader("Find a supporting paper for a claim sentence")
    search_collection = _collection_picker("Collection", key="search_collection_filter")
    sentence = st.text_area("Claim sentence", key="search_sentence")
    top_n = st.slider("Top N", 1, 10, 5, key="search_top_n")
    if st.button("Search library"):
        papers = library.load_library()
        if search_collection != "All":
            papers = [p for p in papers if p.collection == search_collection]
        if not papers:
            st.warning("No papers in this collection. Add papers in the Library tab first.")
        elif not sentence.strip():
            st.warning("Enter a sentence to search for.")
        else:
            with st.spinner("Scoring library candidates..."):
                results = asyncio.run(find_supporting(sentence, papers, api_key=API_KEY, model=MODEL))
            # Kept in session_state (not a local var) so results survive
            # switching to another tab and back -- otherwise the only way to
            # see them again is re-running the search, burning another LLM
            # call for no reason. A fresh claim needs an explicit re-search
            # (button press) same as before; this only avoids losing what
            # was already paid for.
            st.session_state.search_results = results
            st.session_state.search_results_sentence = sentence

    results = st.session_state.get("search_results")
    if results is not None:
        if sentence.strip() and sentence != st.session_state.get("search_results_sentence"):
            st.caption("Showing results for a previous claim — press \"Search library\" again to search this one.")
        if not results:
            st.info("No candidates found.")
        else:
            shown = results[:top_n]
            strong_count = sum(1 for sp in shown if sp.score >= STRONG_MATCH_THRESHOLD)
            if shown and strong_count < len(shown):
                st.caption(
                    f"Only {strong_count} of {len(shown)} result(s) are strong matches "
                    f"(score ≥ {STRONG_MATCH_THRESHOLD}) — the rest are shown for reference but may not "
                    "be reliable citations for this claim."
                )
            for rank, sp in enumerate(shown, start=1):
                if sp.score >= STRONG_MATCH_THRESHOLD:
                    badge = ":green[●] strong"
                elif sp.score >= WEAK_MATCH_THRESHOLD:
                    badge = ":orange[●] moderate"
                else:
                    badge = ":red[●] weak"
                with st.container(border=True):
                    st.markdown(f"**[{rank}] score={sp.score:.3f}** {badge} — **{format_intext_citation(sp.paper)}**")
                    st.markdown(format_apa(sp.paper))
                    if sp.evidence_sentence:
                        st.markdown(f"*Evidence:* {sp.evidence_sentence}")
                        section_caption = _evidence_section_caption(sp.paper, sp.evidence_sentence)
                        if section_caption:
                            st.caption(section_caption)
                        if looks_like_recitation(sp.evidence_sentence):
                            st.caption(
                                "⚠️ This sentence appears to cite another source within it — "
                                "check the original before citing this paper for the claim."
                            )
                    if sp.rationale:
                        st.markdown(f"*Rationale:* {sp.rationale}")

with tab_verify:
    st.subheader("Check whether citing a specific paper here is appropriate")
    verify_collection = _collection_picker("Collection", key="verify_collection_filter")
    papers = library.load_library()
    if verify_collection != "All":
        papers = [p for p in papers if p.collection == verify_collection]
    verify_sentence = st.text_area("Claim sentence", key="verify_sentence")
    if not papers:
        st.warning("No papers in this collection. Add papers in the Library tab first.")
    else:
        titles = [p.title for p in papers]
        selected_title = st.selectbox("Paper to verify against", titles)
        if st.button("Verify citation"):
            if not verify_sentence.strip():
                st.warning("Enter a sentence to verify.")
            else:
                paper = next(p for p in papers if p.title == selected_title)
                with st.spinner("Judging attribution..."):
                    result = asyncio.run(
                        verify_citation(verify_sentence, paper, api_key=API_KEY, model=MODEL)
                    )
                attribution = _attribution_from_rationale(result.rationale)
                color = ATTRIBUTION_COLORS.get(attribution, "gray")
                st.markdown(f":{color}[**{(attribution or 'unknown').upper()}**] — score={result.score:.3f}")
                if result.evidence_sentence:
                    st.markdown(f"*Evidence:* {result.evidence_sentence}")
                    section_caption = _evidence_section_caption(paper, result.evidence_sentence)
                    if section_caption:
                        st.caption(section_caption)
                    if looks_like_recitation(result.evidence_sentence):
                        st.caption(
                            "⚠️ This sentence appears to cite another source within it — "
                            "check the original before citing this paper for the claim."
                        )
                if result.rationale:
                    st.markdown(f"*Rationale:* {result.rationale}")

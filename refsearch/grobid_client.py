import re
import xml.etree.ElementTree as ET

import httpx

from refsearch.models import Paper

_TEI_NS = "http://www.tei-c.org/ns/1.0"
_NS = {"tei": _TEI_NS}


class GrobidUnavailable(RuntimeError):
    pass


async def is_alive(client: httpx.AsyncClient, grobid_url: str = "http://localhost:8070") -> bool:
    try:
        response = await client.get(f"{grobid_url}/api/isalive", timeout=5)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


async def process_pdf(client: httpx.AsyncClient, pdf_path: str, *, grobid_url: str = "http://localhost:8070") -> str:
    """Uploads a PDF to a running GROBID server and returns the TEI XML response
    (full-text extraction: header metadata + body paragraphs).

    consolidateHeader=1 has GROBID cross-check its own layout-parsed header
    against CrossRef (using whatever title/DOI it already found) and use the
    matched record's metadata instead where available. This matters more
    than it sounds: GROBID's raw byline parser can badly mis-segment author
    names when a PDF's author line mixes in affiliations or uses an unusual
    layout (observed on one paper: a 4-author byline came out as 9 garbled
    fragments, one of them literally an affiliation's name) -- CrossRef's
    author list for the same DOI doesn't have that failure mode. Tested
    against a real ingest: fixed the author list, filled in the (previously
    empty) venue too, and added no measurable latency (CrossRef lookup is
    apparently not the bottleneck -- PDF processing itself is)."""
    try:
        with open(pdf_path, "rb") as f:
            response = await client.post(
                f"{grobid_url}/api/processFulltextDocument",
                files={"input": (pdf_path, f, "application/pdf")},
                data={"consolidateHeader": "1"},
                timeout=120,
            )
    except httpx.ConnectError as exc:
        raise GrobidUnavailable(
            f"Could not reach GROBID at {grobid_url}. Start it first, e.g.:\n"
            "  docker run --rm -it -p 8070:8070 grobid/grobid:0.8.1"
        ) from exc
    if response.status_code != 200:
        raise GrobidUnavailable(f"GROBID returned HTTP {response.status_code} for {pdf_path}")
    return response.text


def _text(el: ET.Element | None) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


_MINOR_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "in", "nor", "of",
    "on", "or", "the", "to", "with", "from", "into", "per", "via", "vs",
}


def _smart_title_case(text: str) -> str:
    """PDFs whose cover page typesets the title in all-caps (common for some
    working-paper/journal templates) make GROBID extract it verbatim in all
    caps -- inconsistent with normally-cased titles from other sources. Only
    touches strings that are actually all-caps; normal titles pass through
    unchanged. Not a full AP/Chicago title-case implementation (can't tell an
    embedded acronym like "AI" from a regular word once everything was
    upper-cased -- accepted limitation), just enough to make the library list
    visually consistent."""
    letters = [c for c in text if c.isalpha()]
    if not letters or not all(c.isupper() for c in letters):
        return text
    words = text.lower().split(" ")
    cased = []
    for i, word in enumerate(words):
        if word in _MINOR_WORDS and 0 < i < len(words) - 1:
            cased.append(word)
        else:
            cased.append(word[:1].upper() + word[1:] if word else word)
    return " ".join(cased)


def _extract_title(root: ET.Element) -> str:
    title_el = root.find(".//tei:teiHeader//tei:titleStmt/tei:title", _NS)
    return _smart_title_case(_text(title_el))


def _extract_authors(root: ET.Element) -> list[str]:
    authors = []
    seen: set[str] = set()
    for pers_name in root.findall(
        ".//tei:teiHeader//tei:sourceDesc//tei:biblStruct//tei:analytic/tei:author/tei:persName", _NS
    ):
        forename = " ".join(_text(el) for el in pers_name.findall("tei:forename", _NS))
        surname = _text(pers_name.find("tei:surname", _NS))
        name = " ".join(p for p in [forename, surname] if p)
        # GROBID occasionally leaks a footnote/affiliation marker (observed:
        # a leading "|") into the forename when the PDF's author line has a
        # superscript marker right next to the name -- strip it, since it's
        # never a real part of the name. This can also produce the same
        # author twice (once with the marker, once without); dedupe by the
        # cleaned name so both collapse into one entry.
        name = name.strip(" |*†‡")
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        authors.append(name)
    return authors


def _extract_year(root: ET.Element) -> int | None:
    date_el = root.find(".//tei:teiHeader//tei:sourceDesc//tei:biblStruct//tei:imprint/tei:date", _NS)
    when = date_el.get("when") if date_el is not None else None
    if when and len(when) >= 4 and when[:4].isdigit():
        return int(when[:4])
    return None


def _extract_venue(root: ET.Element) -> str:
    # Journal/conference-proceedings title lives in monogr/title (biblStruct's
    # analytic/title is the paper's own title, not the venue). Preprints (e.g.
    # arXiv-only papers) legitimately have no monogr/title -- empty is correct
    # there, not a parsing failure.
    venue_el = root.find(".//tei:teiHeader//tei:sourceDesc//tei:biblStruct/tei:monogr/tei:title", _NS)
    return _smart_title_case(_text(venue_el))


def _extract_doi(root: ET.Element) -> str | None:
    idno_el = root.find(".//tei:teiHeader//tei:sourceDesc//tei:biblStruct//tei:idno[@type='DOI']", _NS)
    return _text(idno_el) or None


def _extract_abstract(root: ET.Element) -> str:
    abstract_el = root.find(".//tei:teiHeader//tei:profileDesc/tei:abstract", _NS)
    if abstract_el is None:
        return ""
    paragraphs = [_text(p) for p in abstract_el.findall(".//tei:p", _NS)]
    return " ".join(p for p in paragraphs if p) or _text(abstract_el)


def _extract_full_text(root: ET.Element) -> str:
    body_el = root.find(".//tei:text/tei:body", _NS)
    if body_el is None:
        return ""
    paragraphs = [_text(p) for p in body_el.findall(".//tei:p", _NS)]
    return "\n".join(p for p in paragraphs if p)


_PLACEHOLDER_TITLES = {
    "untitled", "untitled document", "untitled-1", "unknown", "no title",
    "document", "new document", "microsoft word", "default",
}


def _is_plausible_title(text: str) -> bool:
    """Rejects /Title metadata that isn't actually a title. Observed in the
    wild: authoring-tool placeholders left unset ("untitled"), and internal
    system/tracking codes some publisher PDF pipelines (e.g. EBSCO exports)
    stamp into the field instead ("1ld;01dec98") -- neither should end up
    displayed as the paper's title. A real title is essentially always
    multiple words with mostly alphabetic content; single short tokens and
    strings dominated by digits/punctuation are the two failure modes seen,
    so both are rejected here rather than passed through verbatim."""
    t = text.strip()
    if not t:
        return False
    if t.lower() in _PLACEHOLDER_TITLES:
        return False
    if len(t.split()) < 2 and len(t) < 20:
        return False
    letters = sum(c.isalpha() for c in t)
    return letters >= len(t) * 0.5


def extract_metadata_title(pdf_path: str) -> str:
    """Falls back to the PDF's own /Title metadata (set by the authoring tool
    -- Word, LaTeX, Acrobat, etc. -- at export time) when GROBID's layout-based
    header model finds nothing. GROBID reads visual layout on the page, which
    can miss a title that's stylistically unusual; the embedded metadata is a
    separate, often-more-reliable source that costs nothing extra to check.
    Returns "" (letting the caller fall through to its next fallback) when the
    metadata value doesn't look like a real title -- see _is_plausible_title."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        title = (reader.metadata.title or "").strip() if reader.metadata else ""
        if not _is_plausible_title(title):
            return ""
        return _smart_title_case(title)
    except Exception:
        return ""


_AUTHOR_SPLIT_RE = re.compile(r"\s*(?:,| and |&|;)\s*")


def _is_plausible_author(name: str) -> bool:
    """Same rationale as _is_plausible_title: some PDF pipelines stamp an
    internal username/system code into /Author (observed: "bos") instead of a
    real name. A single all-lowercase token with no separators reads as a
    code, not a name; genuine single-word author metadata is capitalized."""
    n = name.strip()
    if len(n) < 2:
        return False
    if " " not in n and not (n[:1].isupper() and n.isalpha()):
        return False
    return True


def extract_metadata_authors(pdf_path: str) -> list[str]:
    """Falls back to the PDF's own /Author metadata when GROBID's header model
    finds no authors. Splits on common multi-author delimiters and drops
    entries that don't look like plausible names (see _is_plausible_author)."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        raw = (reader.metadata.author or "").strip() if reader.metadata else ""
        if not raw:
            return []
        candidates = [a.strip() for a in _AUTHOR_SPLIT_RE.split(raw) if a.strip()]
        return [a for a in candidates if _is_plausible_author(a)]
    except Exception:
        return []


_PDF_DATE_YEAR_RE = re.compile(r"D:(\d{4})")


def extract_metadata_year(pdf_path: str) -> int | None:
    """Falls back to the PDF's /CreationDate metadata when GROBID finds no
    imprint date. This is when the file was exported, not necessarily the
    publication year, but for most working papers/preprints (which is what
    GROBID tends to miss a date for) it's a close proxy and beats leaving the
    field blank -- editable by hand afterwards if it's off."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        if not reader.metadata:
            return None
        raw = reader.metadata.get("/CreationDate") or ""
        m = _PDF_DATE_YEAR_RE.match(raw)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def parse_tei(tei_xml: str) -> Paper:
    root = ET.fromstring(tei_xml)
    return Paper(
        title=_extract_title(root),
        authors=_extract_authors(root),
        year=_extract_year(root),
        venue=_extract_venue(root),
        doi=_extract_doi(root),
        abstract=_extract_abstract(root),
        full_text=_extract_full_text(root),
        source="library",
    )

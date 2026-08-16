import re

from refsearch.models import Paper

_WORD_RE = re.compile(r"[A-Za-z]+")

# Heuristic detectors for an in-text citation embedded in a sentence, e.g.
# "(Smith, 2020)", "Smith and Jones (2020)", "[12]" -- used to flag when an
# evidence sentence pulled from a paper's body text is itself reporting a
# *different* source's finding (common in literature-review passages) rather
# than the paper's own claim. Citing the paper we found the sentence in would
# then misattribute the claim to the wrong source. This is intentionally a
# surface-level regex check, not a real citation-graph resolution (we don't
# parse each paper's bibliography) -- it will miss unusual citation styles
# and can false-positive on an incidental "(Word, 1999)"-shaped parenthetical,
# but common APA/numeric styles cover the large majority of cases in practice.
_RECITATION_PATTERNS = [
    # Parenthetical APA: "(Smith, 2020)", "(Smith & Jones, 2020)", "(Smith et al., 2020)"
    re.compile(r"\([A-Z][A-Za-z\-]+(?:\s*(?:&|and|,)\s*[A-Z][A-Za-z\-]+)*(?:\set al\.)?,?\s(?:19|20)\d{2}[a-z]?\)"),
    # Narrative APA: "Smith (2020)", "Smith and Jones (2020)", "Smith et al. (2020)"
    re.compile(r"\b[A-Z][A-Za-z\-]+(?:\s(?:and|&)\s[A-Z][A-Za-z\-]+|\set al\.)?\s\((?:19|20)\d{2}[a-z]?\)"),
    # Numeric bracket style: "[12]", "[3, 4]", "[5-7]"
    re.compile(r"\[\d+(?:\s*[,\-–]\s*\d+)*\]"),
]


def looks_like_recitation(sentence: str) -> bool:
    """True if `sentence` appears to contain an embedded in-text citation of
    another work, rather than being the paper's own original statement --
    see module-level note above on the detection approach and its limits."""
    return any(p.search(sentence) for p in _RECITATION_PATTERNS)


def _apa_authors(authors: list[str]) -> str:
    def to_last_first(name: str) -> str:
        parts = name.strip().split()
        if len(parts) < 2:
            return name
        last = parts[-1]
        initials = " ".join(f"{p[0]}." for p in parts[:-1])
        return f"{last}, {initials}"

    formatted = [to_last_first(a) for a in authors if a.strip()]
    if not formatted:
        return ""
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) <= 20:
        return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"
    return ", ".join(formatted[:19]) + f", ... {formatted[-1]}"


def format_intext_authors(authors: list[str]) -> str:
    """APA in-text style: "Lastname", "Lastname1 and Lastname2", or
    "Lastname1 et al." for 3+ -- for scanning a reference list quickly, the
    way you'd actually cite it in a manuscript, not the full author list."""
    last_names = [a.strip().split()[-1] for a in authors if a.strip()]
    if not last_names:
        return "Unknown"
    if len(last_names) == 1:
        return last_names[0]
    if len(last_names) == 2:
        return f"{last_names[0]} and {last_names[1]}"
    return f"{last_names[0]} et al."


def format_intext_citation(paper: Paper) -> str:
    """APA narrative in-text citation, e.g. "Wen and Zhu (2019)" -- what you'd
    paste directly into a manuscript sentence ("As Wen and Zhu (2019) show,
    ..."), as opposed to format_apa's full reference-list entry."""
    year = paper.year if paper.year else "n.d."
    return f"{format_intext_authors(paper.authors)} ({year})"


def format_apa(paper: Paper) -> str:
    authors = _apa_authors(paper.authors)
    year = paper.year if paper.year else "n.d."
    title = paper.title.strip().rstrip(".")
    venue = paper.venue.strip()
    pieces = [p for p in [authors, f"({year})."] if p]
    citation = f"{pieces[0]} {pieces[1]} " if authors else f"({year}). "
    citation += f"{title}."
    if venue:
        citation += f" {venue}."
    if paper.doi:
        citation += f" https://doi.org/{paper.doi}"
    elif paper.url:
        citation += f" {paper.url}"
    return citation


def bibtex_key(paper: Paper) -> str:
    first_author = paper.authors[0] if paper.authors else "unknown"
    last_name = first_author.strip().split()[-1] if first_author.strip() else "unknown"
    last_name = re.sub(r"[^A-Za-z]", "", last_name).lower() or "unknown"
    year = str(paper.year) if paper.year else "nd"
    title_words = _WORD_RE.findall(paper.title)
    first_word = title_words[0].lower() if title_words else "paper"
    return f"{last_name}{year}{first_word}"


def format_bibtex(paper: Paper) -> str:
    key = bibtex_key(paper)
    entry_type = "misc" if paper.source == "arxiv" else "article"
    authors_bib = " and ".join(paper.authors) if paper.authors else "Unknown"
    fields = [
        f"  author = {{{authors_bib}}}",
        f"  title = {{{paper.title}}}",
        f"  year = {{{paper.year if paper.year else 'n.d.'}}}",
    ]
    if paper.venue:
        field_name = "journal" if entry_type == "article" else "note"
        fields.append(f"  {field_name} = {{{paper.venue}}}")
    if paper.doi:
        fields.append(f"  doi = {{{paper.doi}}}")
    if paper.url:
        fields.append(f"  url = {{{paper.url}}}")
    body = ",\n".join(fields)
    return f"@{entry_type}{{{key},\n{body}\n}}"

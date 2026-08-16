import re

# grobid_client._extract_full_text embeds one of these before each section's
# paragraphs (GROBID's TEI <div><head> structure) so a sentence's section can
# be recovered later. Kept out of the plain body text everywhere else
# (embedding, sentence splitting, the LLM prompt) via strip_markers -- only
# find_section below reads them.
_MARKER_RE = re.compile(r"\[\[SECTION::(.*?)\]\]")


def build_marker(heading: str) -> str:
    return f"[[SECTION::{heading}]]"


def strip_markers(text: str) -> str:
    return _MARKER_RE.sub("", text)


# Keyword-matched against a section heading (case-insensitive substring), not
# a learned classifier -- covers common academic section-naming conventions.
# A heading matching neither list returns None from classify_section rather
# than a guess, since a wrong label here is worse than no label (e.g.
# "Hypothesis Development" sections often both restate prior theory *and*
# state the paper's own predictions -- genuinely ambiguous, deliberately left
# unclassified rather than picked either way).
_BACKGROUND_KEYWORDS = [
    "introduction", "literature review", "related work", "related literature",
    "background", "prior research", "prior studies", "prior literature",
    "theoretical background",
]
_OWN_CONTENT_KEYWORDS = [
    "result", "finding", "discussion", "contribution", "conclusion",
    "implication", "empirical", "analysis", "estimation", "robustness",
    "hypothes",
]


def classify_section(heading: str) -> str | None:
    """"background" = likely reciting prior work to set up the paper
    (elevated re-citation risk if cited as this paper's own claim);
    "own_content" = the paper's own results/discussion/methods; None =
    heading didn't match either list, or is empty."""
    if not heading:
        return None
    h = heading.lower()
    if any(k in h for k in _BACKGROUND_KEYWORDS):
        return "background"
    if any(k in h for k in _OWN_CONTENT_KEYWORDS):
        return "own_content"
    return None


def find_section(full_text: str, sentence: str) -> str | None:
    """Returns the raw heading text of the section `sentence` was found in
    (the nearest preceding marker in `full_text`), or None if the sentence
    isn't present or `full_text` has no markers at all -- e.g. an
    abstract-only candidate, or a library paper ingested before section
    markers were added (needs a re-parse to backfill)."""
    if not sentence:
        return None
    pos = full_text.find(sentence)
    if pos == -1:
        return None
    heading = None
    for m in _MARKER_RE.finditer(full_text):
        if m.start() > pos:
            break
        heading = m.group(1)
    return heading

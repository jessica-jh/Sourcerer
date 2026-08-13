import re

from refsearch.models import Paper

BUSINESS_JOURNALS = [
    "Accounting Review", "Accounting Organizations and Society",
    "Contemporary Accounting Research", "Journal of Accounting and Economics",
    "Journal of Accounting Research", "Review of Accounting Studies",
    "American Economic Review", "Econometrica", "Journal of Political Economy",
    "Quarterly Journal of Economics", "Research Policy",
    "Review of Economic Studies", "Entrepreneurship Theory and Practice",
    "Journal of Business Venturing", "Strategic Entrepreneurship Journal",
    "Journal of Finance", "Journal of Financial and Quantitative Analysis",
    "Journal of Financial Economics", "Review of Finance",
    "Review of Financial Studies", "Human Resource Management",
    "Information Systems Research", "Journal of Management Information Systems",
    "MIS Quarterly", "Journal of International Business Studies",
    "Academy of Management Annals", "Academy of Management Journal",
    "Academy of Management Review", "Administrative Science Quarterly",
    "Harvard Business Review", "Journal of Management",
    "Journal of Management Studies", "Management Science",
    "MIT Sloan Management Review", "Strategic Management Journal",
    "Journal of Consumer Psychology", "Journal of Consumer Research",
    "Journal of Marketing", "Journal of Marketing Research",
    "Journal of the Academy of Marketing Science", "Marketing Science",
    "Journal of Operations Management",
    "Manufacturing & Service Operations Management", "Operations Research",
    "Production and Operations Management", "Organization Science",
    "Organizational Behavior and Human Decision Processes",
    "Journal of Applied Psychology", "Psychological Science",
    "American Sociological Review", "Journal on Computing",
]

PSYCH_SOC_JOURNALS = [
    "Psychological Bulletin", "Psychological Review", "American Psychologist",
    "Psychological Science", "Annual Review of Psychology",
    "Journal of Personality and Social Psychology", "Journal of Applied Psychology",
    "American Journal of Sociology", "American Sociological Review",
    "Annual Review of Sociology", "Social Forces",
    "Sociological Methods & Research", "Journal of Health and Social Behavior",
]

# Non-ACL clusters matched by venue-name substring (no structural identifier available).
CS_AI_MAIN_VENUES = [
    "NeurIPS", "Neural Information Processing Systems",
    "ICML", "International Conference on Machine Learning",
    "ICLR", "International Conference on Learning Representations",
    "AAAI", "IJCAI", "COLM", "Conference on Language Modeling",
    "KDD", "Knowledge Discovery and Data Mining",
    "SIGIR", "WWW", "The Web Conference", "WSDM",
]

ACL_FAMILY_VENUES = ["ACL", "EMNLP", "NAACL", "TACL", "CoNLL"]
ACL_DOI_PREFIX = "10.18653"
ACL_MAIN_FINDINGS_RE = re.compile(r"\d{4}\.(acl|emnlp|naacl)-(long|short|main|findings)")
CONLL_EXCEPTION_SLUGS = {"conll"}

_ALIAS_TABLE = {
    "j mktg": "journal of marketing",
    "jmr": "journal of marketing research",
    "amj": "academy of management journal",
    "amr": "academy of management review",
    "asq": "administrative science quarterly",
    "smj": "strategic management journal",
    "mis quarterly": "mis quarterly",
    "misq": "mis quarterly",
    "jf": "journal of finance",
    "jfe": "journal of financial economics",
    "qje": "quarterly journal of economics",
    "aer": "american economic review",
    "jpe": "journal of political economy",
}

VENUE_PRESETS = {
    "business": BUSINESS_JOURNALS,
    "psych_soc": PSYCH_SOC_JOURNALS,
    "cs_ai": CS_AI_MAIN_VENUES + ACL_FAMILY_VENUES,
}


def normalize_venue(name: str) -> str:
    if not name:
        return ""
    normalized = name.lower().strip()
    normalized = re.sub(r"[^\w\s&]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return _ALIAS_TABLE.get(normalized, normalized)


_NORMALIZED_PRESETS = {
    preset: {normalize_venue(v) for v in venues}
    for preset, venues in VENUE_PRESETS.items()
}


def _is_acl_family(paper: Paper) -> bool:
    venue_lower = (paper.venue or "").lower()
    return any(fam.lower() in venue_lower for fam in ACL_FAMILY_VENUES)


def is_workshop(paper: Paper) -> bool:
    """Best-effort workshop detection. ACL-family venues use structural
    identifiers (DOI prefix + Anthology ID pattern); CoNLL is exempted since
    it is administratively filed as a workshop despite being an independent,
    peer-reviewed conference. Other clusters fall back to a 'workshop'
    substring match on the venue name (known to be imprecise)."""
    venue_lower = (paper.venue or "").lower()
    if any(slug in venue_lower for slug in CONLL_EXCEPTION_SLUGS):
        return False
    if _is_acl_family(paper):
        if paper.doi and paper.doi.startswith(ACL_DOI_PREFIX):
            anthology_hint = (paper.url or "") + (paper.doi or "")
            if ACL_MAIN_FINDINGS_RE.search(anthology_hint.lower()):
                return False
            return True
        return "workshop" in venue_lower
    return "workshop" in venue_lower


def matches_preset(paper: Paper, preset: str, *, include_workshops: bool = False) -> bool:
    if preset == "none" or preset not in _NORMALIZED_PRESETS:
        return True
    if paper.source == "arxiv" and preset == "cs_ai":
        return True
    if preset == "cs_ai" and not include_workshops and is_workshop(paper):
        return False
    normalized = normalize_venue(paper.venue)
    if not normalized:
        return False
    for candidate in _NORMALIZED_PRESETS[preset]:
        if candidate in normalized or normalized in candidate:
            return True
    return False


def filter_by_preset(papers: list[Paper], preset: str, *, include_workshops: bool = False) -> list[Paper]:
    return [p for p in papers if matches_preset(p, preset, include_workshops=include_workshops)]

import re

_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "is", "are", "was",
    "were", "be", "been", "being", "of", "in", "on", "at", "to", "for",
    "with", "by", "as", "and", "or", "but", "if", "so", "than", "then",
    "it", "its", "their", "our", "we", "they", "which", "who", "whom",
    "such", "can", "could", "may", "might", "will", "would", "should",
    "has", "have", "had", "do", "does", "did", "not", "also", "more",
    "most", "other", "into", "over", "under", "between", "through",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")


def _tokenize(sentence: str) -> list[str]:
    return _WORD_RE.findall(sentence)


def clean_query(sentence: str) -> str:
    tokens = [t for t in _tokenize(sentence) if t.lower() not in _STOPWORDS]
    return " ".join(tokens) if tokens else sentence.strip()


def keyphrase_query(sentence: str, max_terms: int = 8) -> str:
    tokens = [t for t in _tokenize(sentence) if t.lower() not in _STOPWORDS]
    scored = sorted(tokens, key=len, reverse=True)
    top_terms = scored[:max_terms]
    seen = set()
    ordered = [t for t in tokens if t in top_terms and not (t in seen or seen.add(t))]
    return " ".join(ordered) if ordered else sentence.strip()


def base_queries(sentence: str) -> list[str]:
    queries = [clean_query(sentence)]
    kp = keyphrase_query(sentence)
    if kp and kp.lower() != queries[0].lower():
        queries.append(kp)
    return queries

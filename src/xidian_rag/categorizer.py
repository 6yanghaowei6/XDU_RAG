from __future__ import annotations


def categorize(text: str, default: str, category_keywords: dict[str, list[str]]) -> str:
    haystack = text.lower()
    best_category = default
    best_score = 0
    for category, keywords in category_keywords.items():
        score = sum(1 for keyword in keywords if keyword.lower() in haystack)
        if score > best_score:
            best_category = category
            best_score = score
    return best_category


def keyword_score(query: str, text: str) -> float:
    tokens = [token for token in query.strip().lower().split() if token]
    if not tokens:
        tokens = list(query.strip().lower())
    if not tokens:
        return 0.0
    haystack = text.lower()
    hits = sum(1 for token in tokens if token and token in haystack)
    return hits / max(len(tokens), 1)

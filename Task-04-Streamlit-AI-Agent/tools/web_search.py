import re

from ddgs import DDGS
from langchain_core.tools import tool

# Try "auto" first, then individual backends. If one fails (DNS/timeout), the next is tried.
BACKENDS = ["auto", "duckduckgo", "yahoo", "mojeek", "bing", "google", "brave"]

# Words removed when the full question returns no results
STOPWORDS = {
    "who", "what", "when", "where", "which", "how", "why", "is", "are", "was",
    "were", "the", "a", "an", "of", "in", "on", "to", "for", "and", "does", "do",
    "did", "current", "currently", "right", "now", "please", "tell", "me", "about",
}


def _shorten(query: str) -> str:
    """'who is the CEO of Anthropic' -> 'CEO Anthropic'"""
    words = re.findall(r"[\w'-]+", query)
    kept = [w for w in words if w.lower() not in STOPWORDS]
    return " ".join(kept) or query


def _format(results) -> str:
    return "\n\n".join(
        f"[{i}] {r.get('title', '')}\n{r.get('body', '')}\nURL: {r.get('href', '')}"
        for i, r in enumerate(results, 1)
    )


@tool
def web_search(query: str) -> str:
    """Search the web in real time for current information, news, facts, prices or
    exchange rates. Input is a plain search query."""
    errors = []
    reached_internet = False  # becomes True if at least one backend responded without a network error

    # Try the full query first, then a shortened keyword version
    for q in dict.fromkeys([query, _shorten(query)]):
        for backend in BACKENDS:
            try:
                results = DDGS(timeout=8).text(q, max_results=5, backend=backend)
            except Exception as exc:
                msg = str(exc)
                # "No results found" is not a network failure
                if "no results" in msg.lower():
                    reached_internet = True
                else:
                    errors.append(f"{backend}: {msg[:60]}")
                continue
            reached_internet = True
            if results:
                return _format(results)

    if reached_internet:
        return "No results found for this query. Try rephrasing it with different keywords."

    return (
        "SEARCH_UNAVAILABLE: all search backends failed (network/DNS problem). "
        "Do not retry. Tell the user the search is unavailable. "
        f"Details: {'; '.join(errors)}"
    )
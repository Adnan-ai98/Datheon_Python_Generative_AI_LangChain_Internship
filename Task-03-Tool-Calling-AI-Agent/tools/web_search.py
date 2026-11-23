from ddgs import DDGS
from langchain_core.tools import tool

# Ek backend fail ho (DNS/timeout) to agla try hota hai
BACKENDS = ["duckduckgo", "bing", "yahoo", "google", "mojeek", "brave"]


@tool
def web_search(query: str) -> str:
    """Search the web in real time for current information, news, facts, prices or
    exchange rates. Input is a plain search query."""
    errors = []
    for backend in BACKENDS:
        try:
            results = DDGS(timeout=8).text(query, max_results=5, backend=backend)
        except Exception as exc:
            errors.append(f"{backend}: {str(exc)[:60]}")
            continue
        if results:
            return "\n\n".join(
                f"[{i}] {r.get('title', '')}\n{r.get('body', '')}\nURL: {r.get('href', '')}"
                for i, r in enumerate(results, 1)
            )
    return (
        "SEARCH_UNAVAILABLE: all search backends failed (network/DNS problem). "
        "Do not retry. Tell the user the search is unavailable. "
        f"Details: {'; '.join(errors)}"
    )

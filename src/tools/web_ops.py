"""Basic web / HTTP tools.

Includes:
    fetch_url  - simple HTTP GET retrieval
    duck_search - DuckDuckGo search (lightweight wrapper)
"""
from __future__ import annotations

import json, textwrap
from typing import List

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from langchain.tools import StructuredTool
    from src.tools.registry import register_tool
    LANGCHAIN_AVAILABLE = True
except Exception:  # pragma: no cover
    LANGCHAIN_AVAILABLE = False
    from src.tools.registry import register_tool


def fetch_url(url: str, max_chars: int = 4000) -> str:
    """Fetch a URL and return truncated text content."""
    if requests is None:
        return "❌ 'requests' not installed. Add it to requirements first."
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        text = r.text
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"
        return text
    except Exception as e:  # pragma: no cover
        return f"❌ fetch failed: {e}"


def duck_search(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo and return a concise list of results.

    Each line: * Title - URL\n  Snippet
    """
    try:
        # Preferred fast API (library)
        from duckduckgo_search import DDGS  # type: ignore
        results: List[dict] = []
        with DDGS() as ddgs:  # pragma: no branch
            for r in ddgs.text(query, max_results=max_results):  # type: ignore
                if not r:
                    continue
                results.append(r)
                if len(results) >= max_results:
                    break
        if not results:
            return "No results."  # pragma: no cover
        lines = []
        for r in results:
            title = r.get("title") or r.get("heading") or "(no title)"
            href = r.get("href") or r.get("url") or ""
            body = (r.get("body") or r.get("snippet") or "").strip()
            lines.append(f"* {title} - {href}\n  {body[:220]}".rstrip())
        return "\n".join(lines)
    except Exception:  # pragma: no cover
        # Fallback: simple HTML query (very coarse)
        if requests is None:
            return "❌ duck_search unavailable: 'duckduckgo_search' lib missing and 'requests' not installed."
        try:
            q = query.replace(" ", "+")
            resp = requests.get(f"https://duckduckgo.com/html/?q={q}", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            resp.raise_for_status()
            import re
            # crude extraction of results
            matches = re.findall(r'<a[^>]*class="result__a"[^>]*>(.*?)</a>', resp.text)[:max_results]
            cleaned = []
            for m in matches:
                # strip tags
                t = re.sub(r"<.*?>", "", m)
                cleaned.append(f"* {t}")
            return "\n".join(cleaned) if cleaned else "No results (fallback)."
        except Exception as e:  # pragma: no cover
            return f"❌ duck_search failed: {e}"


if LANGCHAIN_AVAILABLE:
    register_tool(StructuredTool.from_function(fetch_url, name="fetch_url", description="HTTP GET a URL (truncated)."))
    register_tool(StructuredTool.from_function(duck_search, name="duck_search", description="Search DuckDuckGo (concise results)."))
else:
    class _FetchTool:
        name = "fetch_url"
        description = "HTTP GET a URL (truncated)."
        def __call__(self, url: str, max_chars: int = 4000):  # pragma: no cover
            return fetch_url(url, max_chars=max_chars)
    class _DuckTool:
        name = "duck_search"
        description = "Search DuckDuckGo (concise results)."
        def __call__(self, query: str, max_results: int = 5):  # pragma: no cover
            return duck_search(query, max_results=max_results)
    register_tool(_FetchTool())
    register_tool(_DuckTool())

__all__ = ["fetch_url", "duck_search"]

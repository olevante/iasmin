import os
from typing import Iterable, List

import requests
from bs4 import BeautifulSoup


USER_AGENT = os.getenv("USER_AGENT", "IAsmin-SAP-WebSearch/1.0")


def search_searxng(
    query: str,
    searxng_url: str,
    domains: Iterable[str] | None = None,
    max_results: int = 5,
    timeout: int = 20,
) -> List[dict]:
    headers = {"User-Agent": USER_AGENT}
    base_url = searxng_url.rstrip("/")

    # Tenta JSON primeiro
    params = {
        "q": query,
        "format": "json",
        "language": "pt-BR",
        "safesearch": 1,
    }
    results = []
    try:
        resp = requests.get(base_url, params=params, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
    except Exception:
        results = []

    # Fallback HTML (quando JSON retorna 403/erro)
    if not results:
        params = {
            "q": query,
            "language": "pt-BR",
            "safesearch": 1,
        }
        resp = requests.get(base_url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for art in soup.select("article.result"):
            a = art.select_one("a.result__a")
            if not a or not a.get("href"):
                continue
            title = a.get_text(strip=True)
            url = a["href"]
            content = ""
            snippet = art.select_one(".result__snippet")
            if snippet:
                content = snippet.get_text(strip=True)
            results.append({"title": title, "url": url, "content": content})

    if domains:
        domains = set(d.lower() for d in domains)
        filtered = []
        for r in results:
            url = (r.get("url") or "").lower()
            if any(d in url for d in domains):
                filtered.append(r)
        results = filtered or results

    return results[:max_results]

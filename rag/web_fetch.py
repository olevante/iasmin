import os
from typing import Optional

import requests
from bs4 import BeautifulSoup


USER_AGENT = os.getenv("USER_AGENT", "IAsmin-SAP-WebFetch/1.0")


def fetch_url_text(url: str, max_chars: int = 6000, timeout: int = 25) -> Optional[str]:
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        if resp.status_code >= 400:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        if not text:
            return None
        return text[:max_chars]
    except Exception:
        return None

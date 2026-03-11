import re
from typing import Optional

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound


def _video_id(url: str) -> Optional[str]:
    patterns = [
        r"v=([A-Za-z0-9_\-]{6,})",
        r"youtu\.be/([A-Za-z0-9_\-]{6,})",
        r"youtube\.com/shorts/([A-Za-z0-9_\-]{6,})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def fetch_youtube_transcript(url: str, languages: list[str] | None = None, max_chars: int = 6000) -> Optional[str]:
    vid = _video_id(url)
    if not vid:
        return None
    langs = languages or ["pt", "pt-BR", "en"]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(vid, languages=langs)
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception:
        return None
    text = " ".join([t.get("text", "") for t in transcript]).strip()
    if not text:
        return None
    return text[:max_chars]

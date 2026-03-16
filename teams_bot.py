import os
from typing import Any

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="IAsmiN Teams Bot API")

RAG_API_URL = os.getenv("IASMIN_API_URL", "").strip()


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "bot-api"}


@app.post("/api/messages")
def bot_messages(activity: dict[str, Any]) -> JSONResponse:
    """
    Minimal endpoint expected by Azure Bot/Teams.
    For now, it accepts the activity payload and returns HTTP 200.
    If IASMIN_API_URL is configured, it attempts to call /ask on the RAG API.
    """
    if not RAG_API_URL:
        return JSONResponse({"ok": True, "detail": "Message received"}, status_code=200)

    text = str(activity.get("text", "")).strip()
    user_id = (
        activity.get("from", {}).get("id")
        or activity.get("conversation", {}).get("id")
        or "teams-user"
    )
    if not text:
        return JSONResponse({"ok": True, "detail": "No text in activity"}, status_code=200)

    try:
        resp = requests.post(
            f"{RAG_API_URL.rstrip('/')}/ask",
            json={"question": text, "user_id": user_id},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return JSONResponse({"ok": True, "rag_response": data}, status_code=200)
    except Exception as exc:
        return JSONResponse(
            {"ok": False, "detail": f"RAG call failed: {exc}"},
            status_code=200,
        )

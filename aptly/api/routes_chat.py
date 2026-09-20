# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""The POST /chat endpoint: a plain chat with the local model, for testing.

Not part of the resume/JD analysis pipeline — this exists so the frontend's
"Model Chat" tab can send arbitrary prompts to the same Ollama model the rest
of Aptly uses, as a quick way to confirm the model is running and to see how
it behaves outside the structured-extraction prompts.

Run standalone: not applicable — this module only defines a FastAPI
`APIRouter`. Start the app with `uvicorn aptly.api.main:app --reload`, then:

    curl -X POST http://localhost:8000/chat \\
      -H "Content-Type: application/json" \\
      -d '{"messages": [{"role": "user", "content": "Say hello in five words."}]}'
"""

import requests
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from aptly import config
from aptly.api.models import ChatRequest, ChatResponse
from aptly.llm.client import chat

router = APIRouter()

#: How many of the most recent messages are forwarded to the model. The
#: context window is small (`config.OLLAMA_NUM_CTX`), so an unbounded history
#: would eventually push the oldest turns out anyway — trimming here keeps
#: that predictable rather than leaving it to Ollama's context-shift.
MAX_HISTORY_MESSAGES = 12


@router.post("/chat", response_model=ChatResponse)
async def chat_with_model(request: ChatRequest) -> ChatResponse:
    """Get the local model's reply to a conversation.

    The blocking Ollama call runs via `fastapi.concurrency.run_in_threadpool`
    so a slow CPU-only generation doesn't stall other requests. Only the last
    `MAX_HISTORY_MESSAGES` messages are sent to the model.

    Args:
        request: The parsed `ChatRequest` — the conversation so far.

    Returns:
        A `ChatResponse` with the model's reply and the model tag used.

    Raises:
        HTTPException: 400 if the conversation is empty; 502 with a readable
            message if Ollama is unreachable, times out, or returns an error
            (so the UI can show what went wrong instead of a generic 500).
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty.")

    recent = [m.model_dump() for m in request.messages[-MAX_HISTORY_MESSAGES:]]
    try:
        reply = await run_in_threadpool(chat, recent)
    except requests.ConnectionError:
        raise HTTPException(
            status_code=502,
            detail=f"Can't reach Ollama at {config.OLLAMA_HOST}. Is `ollama serve` running?",
        )
    except requests.Timeout:
        raise HTTPException(
            status_code=502,
            detail=f"The model didn't reply within {config.OLLAMA_TIMEOUT_SECONDS}s.",
        )
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Ollama returned an error: {e}")

    return ChatResponse(reply=reply, model=config.OLLAMA_MODEL)

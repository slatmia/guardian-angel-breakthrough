"""
guardian_debug.py
Ultra-minimal FastAPI server to prove the pipeline works.

Endpoints:
  GET  /health
  POST /guardian/analyze
"""

from fastapi import FastAPI
from pydantic import BaseModel
import time

app = FastAPI(title="Guardian DEBUG Server", version="0.0.1")


class AnalyzeRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "message": "Guardian DEBUG server is running",
    }


@app.post("/guardian/analyze")
def guardian_analyze(req: AnalyzeRequest):
    """
    This is intentionally dumb and fast:
    - No hashing
    - No Ollama
    - No torch
    - No external calls
    If this hangs, the problem is NOT the server logic.
    """
    started = time.time()
    text = req.text

    return {
        "ok": True,
        "mode": "DEBUG_STATIC",
        "echo": text,
        "length": len(text),
        "timing_ms": (time.time() - started) * 1000.0,
    }


if __name__ == "__main__":
    import uvicorn

    # Use a separate port so we don't collide with the main server
    uvicorn.run("guardian_debug:app", host="127.0.0.1", port=11437, reload=False)
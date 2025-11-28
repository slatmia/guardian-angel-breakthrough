"""
guardian_server.py
Minimal, stable Guardian Decision Capsule server.

Endpoints:
  GET  /health
  POST /guardian/analyze

Contract:
  Request:
    { "text": "<user message>" }

  Response (minimal baseline):
    {
      "ok": true,
      "mode": "MINIMAL_BASELINE",
      "echo": "<same text>",
      "length": <len(text)>,
      "timing_ms": <float>
    }
"""

from fastapi import FastAPI
from pydantic import BaseModel
import time
import hashlib

# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    ok: bool
    mode: str
    echo: str
    length: int
    timing_ms: float
    scores: dict


# ─────────────────────────────────────────────
# Hash-based Swarm Scoring
# ─────────────────────────────────────────────

def compute_hash_scores(text: str) -> dict:
    """
    Deterministic hash-based scoring - NO external deps, NO hanging.
    Returns emotion/security/quality scores in [0,1] range.
    """
    # Base hash for deterministic randomness
    base_hash = hashlib.sha256(text.encode('utf-8', errors='ignore')).digest()
    
    # Extract different bytes for different scores
    emotion_bytes = int.from_bytes(base_hash[:4], byteorder='big') / 0xFFFFFFFF
    security_bytes = int.from_bytes(base_hash[4:8], byteorder='big') / 0xFFFFFFFF  
    quality_bytes = int.from_bytes(base_hash[8:12], byteorder='big') / 0xFFFFFFFF
    
    # Keyword-based adjustments
    text_lower = text.lower()
    
    # Emotion: negative words push toward 0, positive toward 1
    emotion_score = emotion_bytes
    if any(word in text_lower for word in ['hate', 'stupid', 'angry', 'frustrated']):
        emotion_score *= 0.3  # Strong negative bias
    elif any(word in text_lower for word in ['love', 'great', 'awesome', 'thanks']):
        emotion_score = 0.7 + emotion_score * 0.3  # Positive bias
        
    # Security: SQL injection patterns boost score toward 1 (dangerous)
    security_score = security_bytes
    dangerous_patterns = ['delete from', 'drop table', 'select *', 'union select', 'or 1=1']
    if any(pattern in text_lower for pattern in dangerous_patterns):
        security_score = 0.8 + security_score * 0.2  # High danger
        
    # Quality: length and coherence boost score
    quality_score = quality_bytes
    if len(text) > 100:  # Longer text tends to be higher effort
        quality_score = 0.4 + quality_score * 0.6
    if any(junk in text_lower for junk in ['asdf', 'lorem ipsum']):
        quality_score *= 0.2  # Junk penalty
        
    return {
        'emotion': round(emotion_score, 3),
        'security': round(security_score, 3), 
        'quality': round(quality_score, 3)
    }

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Guardian Decision Capsule (Minimal)",
    version="0.1.0",
)


@app.get("/health")
def health():
    """
    Simple health probe.
    """
    return {
        "status": "healthy",
        "message": "Guardian MINIMAL server is running",
    }


@app.post("/guardian/analyze", response_model=AnalyzeResponse)
def guardian_analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """
    Minimal, guaranteed-fast implementation:
      - No Ollama
      - No torch / numpy
      - No network calls
    Just echoes the text with some metadata.
    """
    started = time.time()

    text = req.text or ""
    length = len(text)
    
    # Compute hash-based swarm scores
    scores = compute_hash_scores(text)
    
    timing_ms = (time.time() - started) * 1000.0

    return AnalyzeResponse(
        ok=True,
        mode="HASH_SWARM",
        echo=text,
        length=length,
        timing_ms=timing_ms,
        scores=scores,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "guardian_server:app",
        host="127.0.0.1",
        port=11436,
        reload=False,
    )
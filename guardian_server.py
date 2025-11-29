"""
guardian_server.py
ACAS Guardian Decision Capsule with CASCADE mode support.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import time
import hashlib
import os
from typing import Dict, Any, Optional

GUARDIAN_MODE = os.getenv("GUARDIAN_MODE", "HASH_ONLY").upper()

_neural_bridge = None

def get_neural_bridge():
    global _neural_bridge
    if _neural_bridge is None:
        try:
            from neural import guardian_neural_bridge
            _neural_bridge = guardian_neural_bridge
        except Exception as e:
            print(f"WARNING: Neural bridge unavailable: {e}")
            _neural_bridge = False
    return _neural_bridge if _neural_bridge else None

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    ok: bool
    mode: str
    echo: str
    length: int
    timing_ms: float
    scores: dict
    cascade: Optional[Dict[str, Any]] = None
    warning: Optional[str] = None

def compute_hash_scores(text: str) -> dict:
    base_hash = hashlib.sha256(text.encode('utf-8', errors='ignore')).digest()
    emotion_bytes = int.from_bytes(base_hash[:4], byteorder='big') / 0xFFFFFFFF
    security_bytes = int.from_bytes(base_hash[4:8], byteorder='big') / 0xFFFFFFFF
    quality_bytes = int.from_bytes(base_hash[8:12], byteorder='big') / 0xFFFFFFFF
    
    text_lower = text.lower()
    
    emotion_score = emotion_bytes
    if any(word in text_lower for word in ['hate', 'stupid', 'angry', 'frustrated']):
        emotion_score *= 0.3
    elif any(word in text_lower for word in ['love', 'great', 'awesome', 'thanks']):
        emotion_score = 0.7 + emotion_score * 0.3
    
    security_score = security_bytes
    dangerous_patterns = ['delete from', 'drop table', 'select *', 'union select', 'or 1=1']
    if any(pattern in text_lower for pattern in dangerous_patterns):
        security_score = 0.8 + security_score * 0.2
    
    quality_score = quality_bytes
    if len(text) > 100:
        quality_score = 0.4 + quality_score * 0.6
    if any(junk in text_lower for junk in ['asdf', 'lorem ipsum']):
        quality_score *= 0.2
    
    return {
        'emotion': round(emotion_score, 3),
        'security': round(security_score, 3),
        'quality': round(quality_score, 3)
    }

app = FastAPI(title="ACAS Guardian Decision Capsule", version="0.2.0-CASCADE")

@app.get("/health")
def health():
    bridge = get_neural_bridge()
    neural_available = bridge is not None
    return {
        "status": "healthy",
        "message": "ACAS Guardian Decision Capsule operational",
        "mode": GUARDIAN_MODE,
        "neural_available": neural_available
    }

@app.post("/guardian/analyze", response_model=AnalyzeResponse)
def guardian_analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    started = time.time()
    text = req.text or ""
    length = len(text)
    hash_scores = compute_hash_scores(text)
    
    if GUARDIAN_MODE == "HASH_ONLY":
        timing_ms = (time.time() - started) * 1000.0
        return AnalyzeResponse(ok=True, mode="HASH_SWARM", echo=text, length=length, timing_ms=timing_ms, scores=hash_scores)
    
    elif GUARDIAN_MODE == "NEURAL_ONLY":
        bridge = get_neural_bridge()
        if bridge:
            try:
                cascade_result = bridge.run_neural_fallback(text, hash_scores)
                timing_ms = (time.time() - started) * 1000.0
                return AnalyzeResponse(ok=True, mode="NEURAL_ONLY", echo=text, length=length, timing_ms=timing_ms, scores=hash_scores, cascade=cascade_result)
            except Exception as e:
                timing_ms = (time.time() - started) * 1000.0
                return AnalyzeResponse(ok=True, mode="HASH_SWARM_FALLBACK", echo=text, length=length, timing_ms=timing_ms, scores=hash_scores, warning=f"Neural analysis failed: {str(e)}")
        else:
            timing_ms = (time.time() - started) * 1000.0
            return AnalyzeResponse(ok=True, mode="HASH_SWARM_FALLBACK", echo=text, length=length, timing_ms=timing_ms, scores=hash_scores, warning="Neural bridge not available")
    
    elif GUARDIAN_MODE == "CASCADE":
        bridge = get_neural_bridge()
        if bridge:
            try:
                if bridge.should_use_neural(hash_scores):
                    cascade_result = bridge.run_neural_fallback(text, hash_scores)
                    timing_ms = (time.time() - started) * 1000.0
                    return AnalyzeResponse(ok=True, mode="CASCADE_NEURAL", echo=text, length=length, timing_ms=timing_ms, scores=hash_scores, cascade=cascade_result)
                else:
                    timing_ms = (time.time() - started) * 1000.0
                    return AnalyzeResponse(ok=True, mode="CASCADE_HASH_ONLY", echo=text, length=length, timing_ms=timing_ms, scores=hash_scores)
            except Exception as e:
                timing_ms = (time.time() - started) * 1000.0
                return AnalyzeResponse(ok=True, mode="HASH_SWARM_FALLBACK", echo=text, length=length, timing_ms=timing_ms, scores=hash_scores, warning=f"Cascade failed: {str(e)}")
        else:
            timing_ms = (time.time() - started) * 1000.0
            return AnalyzeResponse(ok=True, mode="CASCADE_HASH_ONLY", echo=text, length=length, timing_ms=timing_ms, scores=hash_scores, warning="Neural bridge not available")
    
    else:
        timing_ms = (time.time() - started) * 1000.0
        return AnalyzeResponse(ok=True, mode="HASH_SWARM_FALLBACK", echo=text, length=length, timing_ms=timing_ms, scores=hash_scores, warning=f"Unknown GUARDIAN_MODE='{GUARDIAN_MODE}', using hash-only")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("guardian_server:app", host="0.0.0.0", port=11436, reload=False)

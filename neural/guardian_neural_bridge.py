"""
Guardian Neural Bridge (Cascade controller)

Decides when to invoke the neural adapter vs. HASH_SWARM.
"""

from __future__ import annotations
from typing import Dict, Any

from . import neural_swarm_adapter


def should_use_neural(hash_scores: Dict[str, float]) -> bool:
    """
    Decide whether to engage neural analysis based on hash scores.
    
    Strategy: Look for ANOMALOUS patterns, not absolute thresholds
    - Hash scores are pseudo-random baselines
    - Dangerous content shows UNUSUAL COMBINATIONS
    - Normal text tends to have balanced scores
    """
    emotion = hash_scores.get("emotion", 0.5)
    security = hash_scores.get("security", 0.0)
    quality = hash_scores.get("quality", 0.5)

    # Pattern 1: Extremely negative emotion (regardless of other scores)
    if emotion <= 0.15:
        return True  # Always check very negative content
    
    # Pattern 2: High security + medium-low quality combo
    # (SQL injection has quality ~0.3, benign short text has quality <0.1)
    if security >= 0.85 and 0.2 <= quality <= 0.5:
        return True  # Suspicious mid-range quality with high security
    
    # Pattern 3: Multiple extreme values (unusual pattern)
    extreme_count = 0
    if security >= 0.90:  # Very high security
        extreme_count += 1
    if emotion <= 0.20:  # Very low emotion
        extreme_count += 1
    if quality >= 0.85:  # Very high quality
        extreme_count += 1
    
    if extreme_count >= 2:
        return True  # Multiple extremes = unusual
    
    # Default: hash baseline sufficient
    return False


def run_neural_fallback(text: str, hash_scores: Dict[str, float]) -> Dict[str, Any]:
    """
    Call the neural adapter in a SAFE way.

    IMPORTANT:
    - No blocking imports here
    - No unhandled exceptions
    - If anything goes wrong, we return a small warning payload and never break HASH_SWARM.
    """
    try:
        neural_result = neural_swarm_adapter.analyze_text_neural(text)
    except Exception as ex:
        return {
            "engine": "NEURAL_BRIDGE_ERROR",
            "error": str(ex),
            "hash_scores": hash_scores,
        }

    return {
        "engine": "CASCADE",
        "hash_scores": hash_scores,
        "neural": neural_result,
    }

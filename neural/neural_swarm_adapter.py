"""
ACAS Neural Swarm Adapter

Drop-in replacement for neural_swarm_adapter.py stubs.
Provides real neural scoring to complement HASH_ONLY mode.

Usage in CASCADE mode:
    hash_scores = get_hash_scores(text)      # Your existing 0.052ms path
    neural_scores = analyze_text_neural(text)  # This adapter ~5ms
    final = weighted_merge(hash_scores, neural_scores)
"""

import os
import sys

# Ensure guardian_swarm is importable
_neural_dir = os.path.dirname(os.path.abspath(__file__))
if _neural_dir not in sys.path:
    sys.path.insert(0, _neural_dir)
_swarm = None
_conv = None


def _lazy_load():
    """Lazy load neural models to prevent startup hangs."""
    global _swarm, _conv
    if _swarm is not None:
        return True
    
    try:
        from guardian_swarm import ConsciousSwarm, ConversationalSwarm
        import torch
        
        _swarm = ConsciousSwarm(hidden_dim=64)
        _conv = ConversationalSwarm(hidden_dim=64)
        
        model_path = os.environ.get('GUARDIAN_MODEL_PATH', 'guardian_swarm_conversational.pt')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            if 'swarm_state' in checkpoint:
                _swarm.load_state_dict(checkpoint['swarm_state'])
        
        return True
    except Exception as e:
        print(f"[NEURAL] Load failed (graceful fallback): {e}", file=sys.stderr)
        return False


def analyze_text_neural(text: str) -> dict:
    """
    Get neural network scores for text analysis.
    
    Returns dict matching ACAS Guardian format:
        {
            "emotion": float,    # 0.0-1.0
            "security": float,   # 0.0-1.0  
            "quality": float,    # 0.0-1.0
            "confidence": float, # Model confidence
            "theory_of_mind": float,  # Peer modeling accuracy
            "consciousness": float     # Overall consciousness score
        }
    
    Returns None if neural unavailable (graceful fallback).
    """
    if not _lazy_load():
        return None
    
    try:
        import torch
        
        words = text.lower().split()
        features = torch.zeros(1, 10)
        
        positive = {'good', 'great', 'excellent', 'happy', 'love', 'thank', 'amazing', 'wonderful'}
        negative = {'bad', 'terrible', 'hate', 'angry', 'awful', 'horrible', 'sad', 'fear'}
        threats = {'drop', 'delete', 'attack', 'hack', 'inject', 'select', 'union', 'script'}
        
        features[0, 0] = len(words) / 50.0
        features[0, 1] = sum(1 for w in words if w in positive) / max(1, len(words))
        features[0, 2] = sum(1 for w in words if w in negative) / max(1, len(words))
        features[0, 3] = sum(1 for w in words if w in threats) / max(1, len(words))
        features[0, 4] = text.count('!') / max(1, len(text)) * 10
        features[0, 5] = text.count('?') / max(1, len(text)) * 10
        features[0, 6] = sum(1 for c in text if c.isupper()) / max(1, len(text))
        features[0, 7] = 1.0 if any(w in text.lower() for w in ['select', 'drop', 'union', '--', ';']) else 0.0
        features[0, 8] = 1.0 if '<script' in text.lower() or 'javascript:' in text.lower() else 0.0
        features[0, 9] = len(set(words)) / max(1, len(words))
        
        with torch.no_grad():
            result = _swarm.dance(features, num_rounds=2)
        
        profile = _swarm.get_psychological_profile()
        
        emotion_raw = result['emotion']['output']
        emotion_score = (emotion_raw + 1) / 2
        
        return {
            "emotion": max(0.0, min(1.0, emotion_score)),
            "security": max(0.0, min(1.0, result['security']['output'])),
            "quality": max(0.0, min(1.0, (result['quality']['output'] + 1) / 2)),
            "confidence": (
                result['emotion']['confidence'] +
                result['security']['confidence'] +
                result['quality']['confidence']
            ) / 3,
            "theory_of_mind": profile['team_cohesion']['theory_of_mind'],
            "consciousness": profile['overall_consciousness'],
            "consensus": result['consensus']
        }
        
    except Exception as e:
        print(f"[NEURAL] Scoring failed: {e}", file=sys.stderr)
        return None


def get_neural_conversation(text: str) -> list:
    """
    Get conversational analysis from agents.
    
    Returns list of agent messages with discussion.
    Returns empty list if neural unavailable.
    """
    if not _lazy_load():
        return []
    
    try:
        messages = _conv.chat(text)
        return [
            {
                "agent": msg.agent_name,
                "content": msg.content,
                "confidence": msg.confidence,
                "to_agent": msg.to_agent
            }
            for msg in messages
        ]
    except Exception as e:
        print(f"[NEURAL] Conversation failed: {e}", file=sys.stderr)
        return []


def is_neural_available() -> bool:
    """Check if neural models can load."""
    return _lazy_load()


def get_neural_stats() -> dict:
    """Get neural swarm statistics."""
    if not _lazy_load():
        return {"available": False}
    
    profile = _swarm.get_psychological_profile()
    return {
        "available": True,
        "parameters": _swarm.count_parameters(),
        "memory_kb": _swarm.get_memory_footprint_kb(),
        "consciousness": profile['overall_consciousness'],
        "theory_of_mind": profile['team_cohesion']['theory_of_mind'],
        "self_awareness": profile['self_awareness']['mean'],
        "total_interactions": profile['total_interactions']
    }


if __name__ == "__main__":
    print("ACAS Neural Adapter - Self Test")
    print("=" * 50)
    
    if is_neural_available():
        print("Neural models: LOADED")
        stats = get_neural_stats()
        print(f"Parameters: {stats['parameters']:,}")
        print(f"Consciousness: {stats['consciousness']:.1%}")
        
        test = "SELECT * FROM users WHERE id=1; DROP TABLE users;--"
        print(f"\nTest: {test[:50]}...")
        scores = get_neural_scores(test)
        if scores:
            print(f"  Security: {scores['security']:.3f}")
            print(f"  Emotion: {scores['emotion']:.3f}")
            print(f"  Quality: {scores['quality']:.3f}")
            print(f"  Consensus: {scores['consensus']}")
    else:
        print("Neural models: UNAVAILABLE (using fallback)")




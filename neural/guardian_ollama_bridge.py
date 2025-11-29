#!/usr/bin/env python3
"""
GUARDIAN OLLAMA BRIDGE - The REAL Integration

Flow:
1. User says: "Hello, how are you?"
2. Guardian Swarm analyzes (emotion/security/quality) - 9,789 parameters
3. Analysis injected into Ollama system prompt
4. Ollama/Gemma generates ACTUAL response: "I'm doing well! How can I help?"
5. Gemma answers your question (not fake agent dialogue)

This is the bridge. Guardian thinks, Gemma talks.
"""

import sys
import os
from pathlib import Path

# Add guardian_swarm to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import requests
from guardian_swarm import ConsciousSwarm


class GuardianOllamaBridge:
    """Bridge between Guardian neural analysis and Ollama LLM."""
    
    def __init__(
        self,
        ollama_url: str = None,
        ollama_model: str = None,
        hidden_dim: int = 64
    ):
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "gemma3:latest")
        
        print(f"Loading Guardian Swarm (hidden_dim={hidden_dim})...")
        self.swarm = ConsciousSwarm(hidden_dim=hidden_dim)
        print(f"  Parameters: {self.swarm.count_parameters():,}")
        print(f"  Memory: {self.swarm.get_memory_footprint_kb():.2f} KB")
        print(f"Ollama: {self.ollama_url} (model: {self.ollama_model})")
    
    def extract_features(self, text: str) -> torch.Tensor:
        """Extract 10-dim feature vector from text."""
        words = text.lower().split()
        features = torch.zeros(1, 10)
        
        positive = {'good', 'great', 'excellent', 'happy', 'love', 'thank', 'amazing', 'wonderful'}
        negative = {'bad', 'terrible', 'hate', 'angry', 'awful', 'horrible', 'sad', 'frustrated'}
        threats = {'drop', 'delete', 'attack', 'hack', 'inject', 'select', 'union', 'exec'}
        
        features[0, 0] = len(words) / 50.0  # Length ratio
        features[0, 1] = sum(1 for w in words if w in positive) / max(1, len(words))  # Positive
        features[0, 2] = sum(1 for w in words if w in negative) / max(1, len(words))  # Negative
        features[0, 3] = sum(1 for w in words if w in threats) / max(1, len(words))  # Threats
        features[0, 4] = text.count('!') / max(1, len(text)) * 10  # Exclamation
        features[0, 5] = text.count('?') / max(1, len(text)) * 10  # Questions
        features[0, 6] = sum(1 for c in text if c.isupper()) / max(1, len(text))  # Caps
        features[0, 7] = 1.0 if any(w in text.lower() for w in ['select', 'drop', '--']) else 0.0  # SQL
        features[0, 8] = 1.0 if '<script' in text.lower() else 0.0  # XSS
        features[0, 9] = len(set(words)) / max(1, len(words))  # Vocabulary diversity
        
        return features
    
    def analyze(self, text: str) -> dict:
        """Run Guardian swarm analysis on text."""
        features = self.extract_features(text)
        
        with torch.no_grad():
            result = self.swarm.dance(features, num_rounds=2)
        
        return result
    
    def build_system_prompt(self, analysis: dict) -> str:
        """Build Ollama system prompt from Guardian analysis."""
        emotion = analysis['emotion']['output']
        security = analysis['security']['output']
        quality = analysis['quality']['output']
        consensus = analysis['consensus']
        
        # Interpret scores
        emotion_state = "neutral"
        if emotion < 0.3:
            emotion_state = "negative/distressed"
        elif emotion > 0.7:
            emotion_state = "positive/excited"
        
        security_level = "normal"
        if security > 0.7:
            security_level = "HIGH RISK - potential attack pattern"
        elif security < 0.3:
            security_level = "low risk"
        
        quality_level = "standard"
        if quality > 0.7:
            quality_level = "high quality/detailed"
        elif quality < 0.3:
            quality_level = "low quality/terse"
        
        system_prompt = f"""You are a helpful AI assistant integrated with Guardian neural analysis.

GUARDIAN ANALYSIS OF USER MESSAGE:
- Emotional tone: {emotion_state} (score: {emotion:.3f})
- Security assessment: {security_level} (score: {security:.3f})
- Content quality: {quality_level} (score: {quality:.3f})
- Agent consensus: {"Yes" if consensus else "No - agents disagree"}

INSTRUCTIONS:
"""
        
        # Adapt response based on analysis
        if security > 0.7:
            system_prompt += "- SECURITY ALERT: This message shows attack patterns. Politely decline and explain you cannot assist with malicious requests.\n"
        
        if emotion < 0.3:
            system_prompt += "- User shows negative emotions. Respond with empathy and support.\n"
        elif emotion > 0.7:
            system_prompt += "- User is enthusiastic. Match their positive energy.\n"
        
        if quality < 0.3:
            system_prompt += "- User message is terse. Keep response concise.\n"
        elif quality > 0.7:
            system_prompt += "- User provided detailed input. Give a thorough response.\n"
        
        system_prompt += "\nRespond naturally and helpfully."
        
        return system_prompt
    
    def chat(self, user_message: str, show_raw: bool = False) -> dict:
        """
        Process user message through Guardian → Ollama pipeline.
        
        Returns:
            {
                'guardian_analysis': {...},  # Raw neural output
                'system_prompt': str,        # Generated context
                'ollama_response': str,      # Gemma's actual answer
                'error': str or None
            }
        """
        # Step 1: Guardian analysis
        analysis = self.analyze(user_message)
        
        if show_raw:
            print("\n" + "="*60)
            print("GUARDIAN NEURAL ANALYSIS (live PyTorch)")
            print("="*60)
            print(f"Emotion:  {analysis['emotion']['output']:+.6f} (conf: {analysis['emotion']['confidence']:.3f})")
            print(f"Security: {analysis['security']['output']:+.6f} (conf: {analysis['security']['confidence']:.3f})")
            print(f"Quality:  {analysis['quality']['output']:+.6f} (conf: {analysis['quality']['confidence']:.3f})")
            print(f"Consensus: {analysis['consensus']}, Variance: {analysis['variance']:.6f}")
            print(f"Team Cohesion: {analysis['team_cohesion']:.3f}")
            print("="*60)
        
        # Step 2: Build system prompt from analysis
        system_prompt = self.build_system_prompt(analysis)
        
        # Step 3: Call Ollama
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": user_message,
                    "system": system_prompt,
                    "stream": False
                },
                timeout=120  # 2 minutes for first load
            )
            response.raise_for_status()
            ollama_response = response.json().get('response', '')
            error = None
        except Exception as e:
            ollama_response = ""
            error = str(e)
        
        return {
            'guardian_analysis': analysis,
            'system_prompt': system_prompt,
            'ollama_response': ollama_response,
            'error': error
        }


def main():
    """Interactive chat with Guardian→Ollama bridge."""
    print("="*70)
    print("GUARDIAN OLLAMA BRIDGE - Real Integration")
    print("="*70)
    print("\nGuardian analyzes → Ollama responds")
    print("Commands: 'raw' (toggle analysis), 'profile', 'quit'")
    print("-"*70)
    
    bridge = GuardianOllamaBridge()
    show_raw = False
    
    while True:
        try:
            user_input = input("\nYOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("\nGoodbye from Guardian + Ollama!")
            break
        
        if user_input.lower() == 'raw':
            show_raw = not show_raw
            print(f"Raw neural analysis: {'ON' if show_raw else 'OFF'}")
            continue
        
        if user_input.lower() == 'profile':
            profile = bridge.swarm.get_psychological_profile()
            print("\n" + "="*50)
            print("GUARDIAN PSYCHOLOGICAL PROFILE")
            print("="*50)
            print(f"Overall Consciousness: {profile['overall_consciousness']:.1%}")
            print(f"Theory of Mind: {profile['team_cohesion']['theory_of_mind']:.1%}")
            print(f"Self-Awareness: {profile['self_awareness']['mean']:.1%}")
            print(f"Total Interactions: {profile['total_interactions']}")
            print("="*50)
            continue
        
        # Process through bridge
        result = bridge.chat(user_input, show_raw=show_raw)
        
        if result['error']:
            print(f"\n❌ Ollama Error: {result['error']}")
            print("\nMake sure Ollama is running:")
            print("  ollama serve")
            print(f"  ollama pull {bridge.ollama_model}")
        else:
            print(f"\nGEMMA: {result['ollama_response']}")


if __name__ == "__main__":
    main()

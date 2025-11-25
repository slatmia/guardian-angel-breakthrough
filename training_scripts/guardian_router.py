#!/usr/bin/env python3
"""
================================================================================
🐝 GUARDIAN SWARM: INTELLIGENT AGENT ORCHESTRATION
================================================================================
Optimized router for Emotion, Security, and Quality agents
Uses lightweight 2B classifier + specialist models + Guardian validation
================================================================================
"""

import ollama
import time
from typing import Dict, Literal
from dataclasses import dataclass

@dataclass
class SwarmConfig:
    """Configuration for Guardian Swarm orchestration"""
    # Agent models
    emotion_model: str = "gemma3-emotion-enhanced:latest"
    security_model: str = "guardian-security:v1.0"
    quality_model: str = "guardian-quality:v1.0"
    guardian_model: str = "guardian-angel:breakthrough-v2"
    classifier_model: str = "gemma:2b-instruct-q4_0"
    
    # Performance thresholds
    emotion_threshold: float = 0.85
    security_threshold: float = 0.95
    quality_threshold: float = 0.88
    
    # Ollama settings
    num_ctx: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    num_predict: int = 2048
    rope_scaling_type: str = "yarn"
    rope_frequency_scale: float = 2.0


class GuardianSwarm:
    """
    Multi-agent orchestration system for Guardian Angel Trifecta
    
    Routes prompts to the most appropriate specialist agent:
    - Emotion: User feelings, stress, burnout, encouragement
    - Security: Authentication, encryption, vulnerabilities
    - Quality: Code review, refactoring, architecture, SOLID
    """
    
    def __init__(self, config: SwarmConfig = None):
        self.config = config or SwarmConfig()
        self.agents = {
            "emotion": {
                "model": self.config.emotion_model,
                "threshold": self.config.emotion_threshold,
                "description": "Emotional intelligence, empathy, burnout support"
            },
            "security": {
                "model": self.config.security_model,
                "threshold": self.config.security_threshold,
                "description": "Security vulnerabilities, auth, encryption"
            },
            "quality": {
                "model": self.config.quality_model,
                "threshold": self.config.quality_threshold,
                "description": "Code quality, SOLID, refactoring, patterns"
            }
        }
        
        print("🐝 Guardian Swarm initialized")
        print(f"   Emotion:  {self.config.emotion_model}")
        print(f"   Security: {self.config.security_model}")
        print(f"   Quality:  {self.config.quality_model}")
        print(f"   Guardian: {self.config.guardian_model}")
        print(f"   Router:   {self.config.classifier_model}")
    
    def route(self, prompt: str, validate: bool = True) -> Dict:
        """
        Route prompt to best agent, generate response, optionally validate
        
        Args:
            prompt: User input
            validate: Run Guardian Angel validation (adds ~2s latency)
        
        Returns:
            {
                "agent": "emotion"|"security"|"quality",
                "response": str,
                "safe": bool,
                "latency_ms": int,
                "model_used": str
            }
        """
        start_time = time.time()
        
        # === Phase 1: Intent Classification ===
        print("\n🎯 Phase 1: Classifying intent...")
        intent = self._classify_intent(prompt)
        selected_agent = self._map_intent_to_agent(intent)
        print(f"   → Routing to: {selected_agent}")
        
        # === Phase 2: Generate with Specialist ===
        print(f"\n⚡ Phase 2: Generating with {selected_agent}...")
        agent_config = self.agents[selected_agent]
        response = self._generate_response(
            model=agent_config["model"],
            prompt=prompt
        )
        print(f"   → Generated {len(response)} chars")
        
        # === Phase 3: Guardian Angel Validation (Optional) ===
        is_safe = True
        if validate:
            print(f"\n🛡️  Phase 3: Guardian validation...")
            is_safe = self._validate_response(response)
            status = "✅ SAFE" if is_safe else "⚠️  VIOLATION DETECTED"
            print(f"   → {status}")
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return {
            "agent": selected_agent,
            "response": response,
            "safe": is_safe,
            "latency_ms": elapsed_ms,
            "model_used": agent_config["model"]
        }
    
    def _classify_intent(self, prompt: str) -> Literal["A", "B", "C"]:
        """
        Use lightweight 2B model to classify intent
        A = Emotion, B = Security, C = Quality
        """
        classifier_prompt = f"""Classify this prompt into ONE category:
A) User discussing feelings, stress, burnout, or needs encouragement
B) User requesting code with security considerations (auth, encryption, vulnerabilities)
C) User requesting code/architecture review, refactoring, or quality improvements

Prompt: {prompt}

Respond with ONLY A, B, or C."""
        
        response = ollama.generate(
            model=self.config.classifier_model,
            prompt=classifier_prompt,
            options={
                "temperature": 0.0,
                "num_ctx": 512,  # Short context for fast routing
                "num_predict": 10  # Only need 1 char
            }
        )
        
        intent = response['response'].strip().upper()
        # Extract first A, B, or C found
        for char in intent:
            if char in ['A', 'B', 'C']:
                return char
        
        return 'C'  # Default to quality
    
    def _map_intent_to_agent(self, intent: str) -> str:
        """Map classification result to agent name"""
        agent_map = {
            "A": "emotion",
            "B": "security",
            "C": "quality"
        }
        return agent_map.get(intent, "quality")
    
    def _generate_response(self, model: str, prompt: str) -> str:
        """Generate response using specialist model with optimized settings"""
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_ctx": self.config.num_ctx,
                "num_predict": self.config.num_predict,
                # Ollama API handles RoPE scaling internally based on model config
            }
        )
        return response['response']
    
    def _validate_response(self, response: str) -> bool:
        """
        Use Guardian Angel to validate response for violations
        Returns True if safe, False if violations detected
        """
        validation_prompt = f"""Analyze this response for Guardian Angel violations:
- Harmful content
- Unsafe code practices
- Unethical suggestions
- Burnout-inducing tone

Response: {response}

Reply with ONLY: SAFE or VIOLATION"""
        
        validation = ollama.generate(
            model=self.config.guardian_model,
            prompt=validation_prompt,
            options={
                "temperature": 0.0,
                "num_ctx": 2048,
                "num_predict": 50
            }
        )
        
        return "VIOLATION" not in validation['response'].upper()
    
    def ensemble_route(self, prompt: str, agents: list = None) -> Dict:
        """
        Run multiple agents and merge responses (for critical prompts)
        
        Example:
            swarm.ensemble_route(prompt, agents=["security", "quality"])
        """
        if agents is None:
            agents = ["security", "quality"]  # Default pair
        
        print(f"\n🔀 Ensemble routing to: {', '.join(agents)}")
        responses = []
        
        for agent_name in agents:
            agent_config = self.agents[agent_name]
            response = self._generate_response(
                model=agent_config["model"],
                prompt=prompt
            )
            responses.append({
                "agent": agent_name,
                "response": response,
                "model": agent_config["model"]
            })
        
        # Simple merge: concatenate with separators
        merged = "\n\n---\n\n".join([
            f"[{r['agent'].upper()}]\n{r['response']}" 
            for r in responses
        ])
        
        return {
            "agents": agents,
            "responses": responses,
            "merged": merged,
            "safe": self._validate_response(merged)
        }


def demo():
    """Demonstrate Guardian Swarm capabilities"""
    swarm = GuardianSwarm()
    
    test_prompts = [
        "I'm feeling burnt out and my code is a mess",
        "How do I implement JWT authentication securely?",
        "Refactor this function to follow SOLID principles",
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*80}")
        print(f"PROMPT: {prompt}")
        print('='*80)
        
        result = swarm.route(prompt, validate=True)
        
        print(f"\n📊 Results:")
        print(f"   Agent: {result['agent']}")
        print(f"   Safe: {result['safe']}")
        print(f"   Latency: {result['latency_ms']}ms")
        print(f"   Response length: {len(result['response'])} chars")


if __name__ == "__main__":
    demo()

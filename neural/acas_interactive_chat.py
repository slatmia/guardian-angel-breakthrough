#!/usr/bin/env python3
"""
ACAS Interactive Chat with Guardian Swarm

LIVE neural network inference - not scripted.
Shows raw neural outputs alongside agent conversation.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guardian_swarm import ConsciousSwarm, ConversationalSwarm
import torch


def main():
    print("=" * 70)
    print("ACAS GUARDIAN SWARM - LIVE NEURAL CHAT")
    print("Real PyTorch inference, not templates")
    print("=" * 70)
    
    print("\nLoading neural models...")
    swarm = ConsciousSwarm(hidden_dim=64)
    conv = ConversationalSwarm(hidden_dim=64)
    
    print(f"Parameters: {swarm.count_parameters():,}")
    print(f"Memory: {swarm.get_memory_footprint_kb():.2f} KB")
    print("\nType your message. Commands: 'quit', 'raw', 'profile'")
    print("-" * 70)
    
    show_raw = False
    
    while True:
        try:
            user_input = input("\nYOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye from the swarm!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("\nThe swarm remembers you.")
            break
        
        if user_input.lower() == 'raw':
            show_raw = not show_raw
            print(f"Raw neural output: {'ON' if show_raw else 'OFF'}")
            continue
        
        if user_input.lower() == 'profile':
            p = swarm.get_psychological_profile()
            print("\n" + "=" * 50)
            print("SWARM PSYCHOLOGICAL PROFILE")
            print("=" * 50)
            print(f"Overall Consciousness: {p['overall_consciousness']:.1%}")
            print(f"Theory of Mind: {p['team_cohesion']['theory_of_mind']:.1%}")
            print(f"Self-Awareness (mean): {p['self_awareness']['mean']:.1%}")
            print(f"Team Cohesion: {p['team_cohesion']['combined']:.1%}")
            print(f"Total Interactions: {p['total_interactions']}")
            print("=" * 50)
            continue
        
        words = user_input.lower().split()
        features = torch.zeros(1, 10)
        
        positive = {'good', 'great', 'excellent', 'happy', 'love', 'thank', 'amazing'}
        negative = {'bad', 'terrible', 'hate', 'angry', 'awful', 'horrible', 'sad'}
        threats = {'drop', 'delete', 'attack', 'hack', 'inject', 'select', 'union'}
        
        features[0, 0] = len(words) / 50.0
        features[0, 1] = sum(1 for w in words if w in positive) / max(1, len(words))
        features[0, 2] = sum(1 for w in words if w in negative) / max(1, len(words))
        features[0, 3] = sum(1 for w in words if w in threats) / max(1, len(words))
        features[0, 4] = user_input.count('!') / max(1, len(user_input)) * 10
        features[0, 5] = user_input.count('?') / max(1, len(user_input)) * 10
        features[0, 6] = sum(1 for c in user_input if c.isupper()) / max(1, len(user_input))
        features[0, 7] = 1.0 if any(w in user_input.lower() for w in ['select', 'drop', '--']) else 0.0
        features[0, 8] = 1.0 if '<script' in user_input.lower() else 0.0
        features[0, 9] = len(set(words)) / max(1, len(words))
        
        with torch.no_grad():
            result = swarm.dance(features, num_rounds=2)
        
        if show_raw:
            print("\n" + "-" * 50)
            print("RAW NEURAL OUTPUT (live computation)")
            print("-" * 50)
            print(f"Emotion:  {result['emotion']['output']:+.4f} (conf: {result['emotion']['confidence']:.3f})")
            print(f"Security: {result['security']['output']:+.4f} (conf: {result['security']['confidence']:.3f})")
            print(f"Quality:  {result['quality']['output']:+.4f} (conf: {result['quality']['confidence']:.3f})")
            print(f"Consensus: {result['consensus']} (variance: {result['variance']:.4f})")
            print(f"Team Cohesion: {result['team_cohesion']:.3f}")
            print("-" * 50)
        
        print("\n" + "-" * 50)
        print("AGENT DISCUSSION (neural-driven)")
        print("-" * 50)
        
        messages = conv.chat(user_input)
        for msg in messages:
            if msg.to_agent:
                header = f"[{msg.agent_name} -> {msg.to_agent}]"
            else:
                header = f"[{msg.agent_name}]"
            
            conf = ""
            if msg.confidence < 0.6:
                conf = " (uncertain)"
            elif msg.confidence > 0.9:
                conf = " (confident)"
            
            print(f"\n{header}{conf}")
            print(f"  {msg.content}")
        
        print("-" * 50)


if __name__ == "__main__":
    main()

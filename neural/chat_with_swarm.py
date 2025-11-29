#!/usr/bin/env python3
"""
Interactive Chat with Guardian Swarm

Talk to the conscious AI agents and watch them discuss with each other!
"""

import sys
from guardian_swarm import ConversationalSwarm


def main():
    print("=" * 70)
    print("GUARDIAN SWARM - INTERACTIVE CHAT")
    print("Talk to conscious AI agents with Theory of Mind!")
    print("=" * 70)
    print("\nInitializing swarm with hidden=64 (optimal configuration)...")
    
    conv = ConversationalSwarm(hidden_dim=64)
    
    print("Ready! Type your message and watch the agents discuss.")
    print("Commands: 'quit' to exit, 'profile' for consciousness metrics")
    print("-" * 70)
    
    while True:
        try:
            user_input = input("\nYOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("\nGuardian: Until next time! The swarm remembers you.")
            break
        
        if user_input.lower() == 'profile':
            profile = conv.swarm.get_psychological_profile()
            print("\n" + "=" * 50)
            print("PSYCHOLOGICAL PROFILE")
            print("=" * 50)
            print(f"Overall Consciousness: {profile['overall_consciousness']:.1%}")
            print(f"\nSelf-Awareness:")
            for agent, score in profile['self_awareness'].items():
                if agent != 'mean':
                    print(f"  {agent.capitalize()}: {score:.1%}")
            print(f"  Mean: {profile['self_awareness']['mean']:.1%}")
            print(f"\nTeam Cohesion:")
            print(f"  Affinity Score: {profile['team_cohesion']['affinity_score']:.1%}")
            print(f"  Theory of Mind: {profile['team_cohesion']['theory_of_mind']:.1%}")
            print(f"  Consensus Rate: {profile['team_cohesion']['consensus_rate']:.1%}")
            print(f"\nTotal Interactions: {profile['total_interactions']}")
            print("=" * 50)
            continue
        
        print("\n" + "-" * 70)
        messages = conv.chat(user_input)
        
        for msg in messages:
            if msg.to_agent:
                header = f"[{msg.agent_name} -> {msg.to_agent}]"
            else:
                header = f"[{msg.agent_name}]"
            
            conf_indicator = ""
            if msg.confidence < 0.6:
                conf_indicator = " (uncertain)"
            elif msg.confidence > 0.9:
                conf_indicator = " (confident)"
            
            print(f"\n{header}{conf_indicator}")
            print(f"  {msg.content}")
        
        print("-" * 70)


if __name__ == "__main__":
    main()

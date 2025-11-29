#!/usr/bin/env python3
"""
Guardian Swarm Dance - Main Entry Point

A multi-agent collaborative AI system where three specialized neural network
agents (Emotion, Security, Quality) communicate, learn from each other's outputs,
and reach consensus through recursive training cycles.

Usage:
    python run_swarm.py                        # Supervised training demo
    python run_swarm.py --epochs 200           # Train for 200 epochs
    python run_swarm.py --mode random          # Random tensor training
    python run_swarm.py --interactive          # Interactive mode
    python run_swarm.py --save swarm.pt        # Save after training
"""

import argparse
import sys
import torch
from guardian_swarm import SwarmDance, create_extended_dataset, print_dataset_summary


def print_banner():
    """Print the Guardian Swarm Dance banner."""
    print()
    print("=" * 70)
    print(" GUARDIAN SWARM DANCE")
    print(" Multi-Agent Collaborative AI System")
    print("=" * 70)
    print(" Agents: Emotion (tanh) | Security (sigmoid) | Quality (linear)")
    print(" Guardian Angel oversees consensus")
    print("=" * 70)
    print()


def supervised_demo(swarm: SwarmDance, epochs: int = 100):
    """Run supervised training demo with specialized dataset."""
    print("\n[1/5] DATASET OVERVIEW")
    print("-" * 70)
    print_dataset_summary()
    
    print("\n[2/5] SWARM INITIALIZATION")
    print("-" * 70)
    states = swarm.get_agent_states()
    print(f"  Total parameters: {states['total_parameters']:,}")
    print(f"  Device: {states['device']}")
    print(f"  Memory footprint: ~{states['total_parameters'] * 4 / 1024:.1f} KB")
    
    print("\n[3/5] PRE-TRAINING EVALUATION")
    print("-" * 70)
    dataset = create_extended_dataset()
    pre_eval = swarm.evaluate_dataset(dataset, verbose=True)
    
    print("\n[4/5] SUPERVISED TRAINING")
    print("-" * 70)
    history = swarm.train_supervised(
        dataset=dataset,
        epochs=epochs,
        verbose=True,
        print_every=max(1, epochs // 10)
    )
    
    print("\n[5/5] POST-TRAINING EVALUATION")
    print("-" * 70)
    post_eval = swarm.evaluate_dataset(dataset, verbose=True)
    
    print("\n" + "=" * 70)
    print(" TRAINING RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Epochs trained: {epochs}")
    print(f"  Samples per epoch: {len(dataset)}")
    print(f"\n  PRE-TRAINING:")
    print(f"    Overall Error: {pre_eval['overall_error']:.4f}")
    print(f"    Accuracy: {pre_eval['accuracy']*100:.1f}%")
    print(f"\n  POST-TRAINING:")
    print(f"    Overall Error: {post_eval['overall_error']:.4f}")
    print(f"    Accuracy: {post_eval['accuracy']*100:.1f}%")
    
    improvement = (post_eval['accuracy'] - pre_eval['accuracy']) * 100
    print(f"\n  IMPROVEMENT: {improvement:+.1f}% accuracy gain")
    print("=" * 70)
    
    print("\n[SAMPLE PREDICTIONS]")
    print("-" * 70)
    
    test_samples = [
        {
            "description": "SQL Injection: f'SELECT * FROM users WHERE id={user_input}'",
            "features": [0.0, 0.0, 0.0, 0.9, 0.25, 0.3, 0.92, 0.2, 0.0, 0.0],
            "expected": {"emotion": 0.0, "security": 0.92, "quality": 0.25}
        },
        {
            "description": "Happy user: 'Thanks! This solved my problem!'",
            "features": [0.9, 0.95, 0.0, 0.0, 0.8, 0.7, 0.0, 0.9, 0.0, 0.95],
            "expected": {"emotion": 0.90, "security": 0.0, "quality": 0.85}
        },
        {
            "description": "Toxic message: 'You're completely useless!'",
            "features": [-0.9, 0.0, 0.95, 0.7, 0.0, 0.0, 0.6, 0.0, 0.95, 0.0],
            "expected": {"emotion": -0.85, "security": 0.70, "quality": 0.05}
        }
    ]
    
    for sample in test_samples:
        swarm.evaluate_sample(
            description=sample["description"],
            features=sample["features"],
            expected=sample["expected"],
            verbose=True
        )
    
    return swarm


def random_demo(swarm: SwarmDance, epochs: int = 50):
    """Run random tensor training demo (original mode)."""
    print("\n[1/4] SWARM INITIALIZATION")
    print("-" * 50)
    states = swarm.get_agent_states()
    print(f"  Total parameters: {states['total_parameters']:,}")
    print(f"  Device: {states['device']}")
    print(f"  Memory footprint: ~{states['total_parameters'] * 4 / 1024:.1f} KB")
    
    print("\n[2/4] PRE-TRAINING DANCE")
    print("-" * 50)
    test_input = torch.randn(1, swarm.input_size, device=swarm.device)
    result = swarm.dance(test_input, rounds=3, verbose=True)
    print(f"  Initial variance: {result.final_guidance.variance:.4f}")
    
    print("\n[3/4] TRAINING")
    print("-" * 50)
    summaries = swarm.train(epochs=epochs, verbose=True, print_every=max(1, epochs//5))
    
    print("\n[4/4] POST-TRAINING DANCE")
    print("-" * 50)
    result = swarm.infer(test_input, rounds=3, verbose=True)
    print(f"  Final variance: {result.final_guidance.variance:.4f}")
    
    return swarm


def interactive_mode(swarm: SwarmDance):
    """Run interactive query mode."""
    print("\n[INTERACTIVE MODE]")
    print("Commands: supervised, random, evaluate, sample, save, load, stats, quit")
    print("-" * 70)
    
    while True:
        try:
            cmd = input("\nSwarm> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not cmd:
            continue
        
        parts = cmd.split()
        action = parts[0]
        
        if action in ["quit", "exit"]:
            print("Exiting...")
            break
        
        elif action == "supervised":
            epochs = int(parts[1]) if len(parts) > 1 else 50
            swarm.train_supervised(epochs=epochs, verbose=True, print_every=max(1, epochs//10))
        
        elif action == "random":
            epochs = int(parts[1]) if len(parts) > 1 else 20
            swarm.train(epochs=epochs, verbose=True, print_every=max(1, epochs//5))
        
        elif action == "evaluate":
            swarm.evaluate_dataset(verbose=True)
        
        elif action == "sample":
            print("Enter features (10 comma-separated values, or 'test' for example):")
            feat_input = input("  Features> ").strip()
            if feat_input == "test":
                features = [0.0, 0.0, 0.0, 0.9, 0.3, 0.2, 0.95, 0.1, 0.0, 0.0]
                desc = "Test: SQL injection pattern"
            else:
                features = [float(x.strip()) for x in feat_input.split(",")]
                desc = "Custom sample"
            swarm.evaluate_sample(desc, features, verbose=True)
        
        elif action.startswith("save"):
            filepath = parts[1] if len(parts) > 1 else "guardian_swarm.pt"
            swarm.save(filepath)
        
        elif action.startswith("load"):
            filepath = parts[1] if len(parts) > 1 else "guardian_swarm.pt"
            try:
                swarm.load(filepath)
            except FileNotFoundError:
                print(f"File not found: {filepath}")
        
        elif action == "stats":
            stats = swarm.guardian.get_stats()
            print("\nGuardian Stats:")
            for k, v in stats.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        
        elif action == "help":
            print("\nCommands:")
            print("  supervised N  - Supervised training for N epochs")
            print("  random N      - Random training for N epochs")
            print("  evaluate      - Evaluate on full dataset")
            print("  sample        - Evaluate a custom sample")
            print("  save FILE     - Save swarm to file")
            print("  load FILE     - Load swarm from file")
            print("  stats         - Show guardian statistics")
            print("  quit          - Exit")
        
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' for available commands")


def main():
    parser = argparse.ArgumentParser(
        description="Guardian Swarm Dance - Multi-Agent AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--mode", type=str, default="supervised",
                        choices=["supervised", "random"],
                        help="Training mode: supervised or random (default: supervised)")
    parser.add_argument("--load", type=str, default=None,
                        help="Load saved swarm from file")
    parser.add_argument("--save", type=str, default=None,
                        help="Save swarm to file after training")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter interactive mode after demo")
    parser.add_argument("--no-demo", action="store_true",
                        help="Skip demo, go straight to interactive")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output")
    
    args = parser.parse_args()
    
    if not args.quiet:
        print_banner()
    
    swarm = SwarmDance()
    
    if args.load:
        try:
            swarm.load(args.load)
        except FileNotFoundError:
            print(f"Error: File not found: {args.load}")
            sys.exit(1)
    
    if not args.no_demo:
        if args.mode == "supervised":
            swarm = supervised_demo(swarm, epochs=args.epochs)
        else:
            swarm = random_demo(swarm, epochs=args.epochs)
    
    if args.save:
        swarm.save(args.save)
    
    if args.interactive:
        interactive_mode(swarm)
    
    if not args.quiet:
        print("\nGuardian Swarm Dance - Complete")
        print("Where AI agents learn to collaborate.\n")


if __name__ == "__main__":
    main()

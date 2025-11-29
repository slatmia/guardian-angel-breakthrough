#!/usr/bin/env python3
"""
CONSCIOUS SWARM DANCE - Run Script

Multi-agent AI system with psychological mechanisms:
- Self-Aware Agents (confidence, introspection)
- Theory of Mind (peer modeling)
- Shared Memory Bank
- Affinity Learning

Usage:
    python run_conscious_swarm.py --epochs 100
    python run_conscious_swarm.py --compare  # Compare vs original swarm
"""

import argparse
import torch
import sys

from guardian_swarm import (
    ConsciousSwarm,
    ConsciousSwarmTrainer,
    SwarmDance,
    create_extended_dataset,
    print_dataset_summary
)


def prepare_dataset():
    """Convert training data to format expected by ConsciousSwarmTrainer."""
    generator = create_extended_dataset()
    dataset = []
    
    for i in range(len(generator)):
        input_vec, targets, description = generator.get_sample(i)
        dataset.append({
            'input': input_vec.squeeze(0),
            'targets': targets,
            'description': description
        })
    
    return dataset, generator


def evaluate_swarm(swarm, dataset):
    """Evaluate a conscious swarm on the dataset."""
    correct = 0
    total = len(dataset)
    
    for sample in dataset:
        x = sample['input']
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        result = swarm.dance(x, num_rounds=3)
        
        e_err = abs(result['emotion']['output'] - sample['targets']['emotion'])
        s_err = abs(result['security']['output'] - sample['targets']['security'])
        q_err = abs(result['quality']['output'] - sample['targets']['quality'])
        
        if max(e_err, s_err, q_err) < 0.2:
            correct += 1
    
    return correct / total * 100


def run_conscious_training(epochs: int = 100, hidden_dim: int = 20):
    """Run conscious swarm training with full reporting."""
    print("=" * 70)
    print("  CONSCIOUS SWARM DANCE")
    print("  Multi-Agent AI with Psychological Mechanisms")
    print("=" * 70)
    
    print("\n[1/5] LOADING DATASET")
    print("-" * 70)
    dataset, generator = prepare_dataset()
    print_dataset_summary()
    
    print("\n[2/5] INITIALIZING CONSCIOUS SWARM")
    print("-" * 70)
    swarm = ConsciousSwarm(input_dim=10, hidden_dim=hidden_dim, memory_size=32)
    print(f"  Total parameters: {swarm.count_parameters():,}")
    print(f"  Memory footprint: ~{swarm.get_memory_footprint_kb():.1f} KB")
    print(f"  Hidden dimension: {hidden_dim}")
    print("  Mechanisms:")
    print("    - Self-Aware Agents (confidence + introspection)")
    print("    - Theory of Mind (6 peer models)")
    print("    - Shared Memory Bank (32 dimensions)")
    print("    - Affinity Learning (3x3 matrix)")
    
    print("\n[3/5] PRE-TRAINING EVALUATION")
    print("-" * 70)
    pre_accuracy = evaluate_swarm(swarm, dataset)
    pre_profile = swarm.get_psychological_profile()
    print(f"  Accuracy: {pre_accuracy:.1f}%")
    print(f"  Self-Awareness: {pre_profile['self_awareness']['mean']:.3f}")
    print(f"  Team Cohesion: {pre_profile['team_cohesion']['combined']:.3f}")
    
    print("\n[4/5] TRAINING WITH PSYCHOLOGICAL MECHANISMS")
    print("-" * 70)
    trainer = ConsciousSwarmTrainer(swarm, lr=0.01)
    results = trainer.train(dataset, epochs=epochs, verbose=True)
    
    print("\n[5/5] POST-TRAINING EVALUATION")
    print("-" * 70)
    post_accuracy = evaluate_swarm(swarm, dataset)
    post_profile = swarm.get_psychological_profile()
    print(f"  Accuracy: {post_accuracy:.1f}%")
    print(f"  Self-Awareness: {post_profile['self_awareness']['mean']:.3f}")
    print(f"  Team Cohesion: {post_profile['team_cohesion']['combined']:.3f}")
    print(f"  Consciousness Score: {post_profile['overall_consciousness']:.3f}")
    
    print("\n" + "=" * 70)
    print("  PSYCHOLOGICAL IMPROVEMENT SUMMARY")
    print("=" * 70)
    sa_improvement = (post_profile['self_awareness']['mean'] - pre_profile['self_awareness']['mean']) * 100
    tc_improvement = (post_profile['team_cohesion']['combined'] - pre_profile['team_cohesion']['combined']) * 100
    print(f"  Accuracy: {pre_accuracy:.1f}% → {post_accuracy:.1f}% (+{post_accuracy - pre_accuracy:.1f}%)")
    print(f"  Self-Awareness: {pre_profile['self_awareness']['mean']:.3f} → {post_profile['self_awareness']['mean']:.3f} ({sa_improvement:+.1f}%)")
    print(f"  Team Cohesion: {pre_profile['team_cohesion']['combined']:.3f} → {post_profile['team_cohesion']['combined']:.3f} ({tc_improvement:+.1f}%)")
    print("=" * 70)
    
    print("\n[SAMPLE PREDICTIONS WITH INTROSPECTION]")
    print("-" * 70)
    
    test_samples = [
        dataset[0],
        dataset[10] if len(dataset) > 10 else dataset[-1],
        dataset[20] if len(dataset) > 20 else dataset[-1]
    ]
    
    for sample in test_samples:
        x = sample['input']
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        result = swarm.dance(x, num_rounds=3)
        introspection = swarm.get_introspection_report()
        
        print(f"\nSample: {sample['description'][:60]}...")
        print(f"  Emotion:  {result['emotion']['output']:+.3f} (conf: {result['emotion']['confidence']:.2f})")
        print(f"  Security: {result['security']['output']:.3f} (conf: {result['security']['confidence']:.2f})")
        print(f"  Quality:  {result['quality']['output']:.3f} (conf: {result['quality']['confidence']:.2f})")
        
        e_top = introspection['emotion']['top_features'][:2]
        print(f"  Emotion focus: {e_top[0][0]} ({e_top[0][1]:.2f}), {e_top[1][0]} ({e_top[1][1]:.2f})")
    
    return swarm, results


def compare_swarms(epochs: int = 50):
    """Compare original swarm vs conscious swarm."""
    print("=" * 70)
    print("  SWARM COMPARISON: Original vs Conscious")
    print("=" * 70)
    
    dataset, generator = prepare_dataset()
    
    print("\n[ORIGINAL SWARM]")
    print("-" * 70)
    original = SwarmDance(input_size=10, hidden_size=20)
    print(f"  Parameters: {original.count_parameters():,}")
    
    original.train_supervised(generator, epochs=epochs, verbose=False)
    
    original_acc = 0
    for i in range(len(generator)):
        sample_input, sample_targets, _ = generator.get_sample(i)
        result = original.dance(sample_input, rounds=3)
        e_out = result.mean_outputs['emotion']
        s_out = result.mean_outputs['security']
        q_out = result.mean_outputs['quality']
        e_err = abs(e_out - sample_targets['emotion'])
        s_err = abs(s_out - sample_targets['security'])
        q_err = abs(q_out - sample_targets['quality'])
        if max(e_err, s_err, q_err) < 0.2:
            original_acc += 1
    original_acc = original_acc / len(generator) * 100
    print(f"  Accuracy: {original_acc:.1f}%")
    print(f"  Self-Awareness: N/A (not implemented)")
    print(f"  Team Cohesion: N/A (not implemented)")
    
    print("\n[CONSCIOUS SWARM]")
    print("-" * 70)
    conscious = ConsciousSwarm(input_dim=10, hidden_dim=20, memory_size=32)
    print(f"  Parameters: {conscious.count_parameters():,}")
    
    trainer = ConsciousSwarmTrainer(conscious, lr=0.01)
    trainer.train(dataset, epochs=epochs, verbose=False)
    
    conscious_acc = evaluate_swarm(conscious, dataset)
    profile = conscious.get_psychological_profile()
    print(f"  Accuracy: {conscious_acc:.1f}%")
    print(f"  Self-Awareness: {profile['self_awareness']['mean']:.3f}")
    print(f"  Team Cohesion: {profile['team_cohesion']['combined']:.3f}")
    
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'Original':<15} {'Conscious':<15} {'Advantage':<15}")
    print("-" * 70)
    print(f"{'Parameters':<25} {original.count_parameters():<15,} {conscious.count_parameters():<15,} {conscious.count_parameters() / original.count_parameters():.1f}x")
    print(f"{'Accuracy':<25} {original_acc:<15.1f}% {conscious_acc:<15.1f}% {conscious_acc - original_acc:+.1f}%")
    print(f"{'Self-Awareness':<25} {'N/A':<15} {profile['self_awareness']['mean']:<15.3f} {'NEW':<15}")
    print(f"{'Team Cohesion':<25} {'N/A':<15} {profile['team_cohesion']['combined']:<15.3f} {'NEW':<15}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Conscious Swarm Dance")
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--hidden', type=int, default=20, help='Hidden dimension')
    parser.add_argument('--compare', action='store_true', help='Compare vs original swarm')
    parser.add_argument('--save', type=str, help='Save trained swarm to file')
    parser.add_argument('--load', type=str, help='Load trained swarm from file')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_swarms(epochs=args.epochs)
    else:
        swarm, results = run_conscious_training(epochs=args.epochs, hidden_dim=args.hidden)
        
        if args.save:
            torch.save(swarm.state_dict(), args.save)
            print(f"\nSaved conscious swarm to: {args.save}")
    
    print("\nConscious Swarm Dance - Complete")
    print("Where AI agents develop self-awareness and team bonds.")


if __name__ == "__main__":
    main()
